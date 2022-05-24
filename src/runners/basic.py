import math
import re
import time
from copy import deepcopy

import pytorch_lightning as pl
from functools import reduce
import src
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torch.distributed as dist
from src.data.datamodule import DataModule
from src.my_typing import *
from src.utils.fn import apply_to_dict, get_coeff_iter, instantiate_no_recursive, reduce_loss, merge_outputs
from src.utils.var_pool import VarPool

from . import log


class BasicRunner(pl.LightningModule):
    metric: torch.nn.ModuleList

    def __init__(self,
                 dm: DataModule,
                 load_from_checkpoint: str = None,
                 test_when_val: bool = True,
                 train_with_dev: bool = False,
                 ignore_punct=None,   # removed, no effect
                 loss_reduction_mode=None  # removed, no effect
                ):
        self.dm: DataModule = dm
        super().__init__()

        self.checkpoint_path = load_from_checkpoint
        self.test_when_val = test_when_val
        self.train_with_dev = train_with_dev
        self.model: BasicModel = instantiate(src.g_cfg.model)

        self._has_setup = False
        self._init_scheduler_when_running = None
        self.save_hyperparameters(OmegaConf.to_container(src.g_cfg))

    def setup(self, stage: Optional[str] = None) -> None:
        if self._has_setup: return

        with open_dict(src.g_cfg.task):  # setup n_words, n_tag, n_rel ...
            src.g_cfg.task = OmegaConf.merge(src.g_cfg.task, self.dm.get_vocab_count())

        self.model.setup(self.dm)
        self.model.set_trainer(self.trainer)
        self.metric = torch.nn.ModuleList([
            instantiate(src.g_cfg.metric, extra_vocab=self.dm.vocabs),
            instantiate(src.g_cfg.metric, extra_vocab=self.dm.vocabs)
        ])
        if self.__class__ is BasicRunner:
            # workaround of loading when use setup.
            # setup is called by trainer automatically, and before setup(), there is no model.
            if self.checkpoint_path:
                self.load_model_inplace(self.checkpoint_path)
            self._has_setup = True

    def forward(self, x, seq_len):
        score = self.model(x, seq_len)
        predict = self.model.decode(score, seq_len)
        return predict

    # def on_fit_start(self):
    #     # log.info(self)
    #     self.model.set_trainer(self.trainer)

    def on_train_start(self):

        self.model.normalize_embedding('begin')
        if (scheduler_cfg := self._init_scheduler_when_running) is not None:
            n_batches = math.ceil(len(self.trainer.train_dataloader) / self.trainer.accumulate_grad_batches)
            resolved_scheduler_args = {}
            for key, value in scheduler_cfg.args.items():
                if isinstance(value, str) and value.endswith(' epoch'):
                    value = int(value.split()[0]) * n_batches
                resolved_scheduler_args[key] = value
            scheduler = instantiate_no_recursive(resolved_scheduler_args, optimizer=self.trainer.optimizers[0])
            scheduler = {
                'scheduler': scheduler,
                'interval': scheduler_cfg.interval,
                'frequency': scheduler_cfg.frequency,
                'monitor': src.g_cfg.watch_field,
                'strict': True
            }
            self.trainer.lr_schedulers = self.trainer.configure_schedulers([scheduler],
                                                                           monitor=None,
                                                                           is_manual_optimization=False)

    def on_train_epoch_start(self):
        self.model.normalize_embedding('epoch')

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self.model.normalize_embedding('batch')

    def training_step(self, batch, batch_idx):
        x, y = batch['train']
        vp = self.model.set_varpool(VarPool(**x, **y))
        self.log('bs', len(x['id']), prog_bar=True, logger=False)
        self.log('mlen', x['seq_len'].max(), prog_bar=True, logger=False)

        score = self.model(x, vp)
        loss = self.model.loss(score, y, vp)

        with torch.no_grad():
            detailed_loss = apply_to_dict(loss[1], lambda x: self.reduce_loss(x, vp))
            self.log_dict(detailed_loss, prog_bar=True, logger=False)
            self.log_dict({f'train/{k}': v for k, v in detailed_loss.items()})
        loss = self.reduce_loss(loss[0], vp)
        self.log('train/sup_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self.metric[0].reset()
        self.metric[1].reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        # y is used for evaluate brat. to show label acc given gold boundary
        # TODO do not forward the whole y.
        vp = self.model.set_varpool(VarPool(**x, **y))
        score = self.model(x, vp)
        loss = self.model.loss(score, y, vp)
        with torch.no_grad():
            detailed_loss = apply_to_dict(loss[1], lambda x: self.reduce_loss(x, vp))
            self.log_dict({f'train/{k}': v for k, v in detailed_loss.items()})
        loss = self.reduce_loss(loss[0], vp).item()
        predict = self.model.decode(score, vp)
        mask = vp.mask
        self.metric[dataloader_idx].update(predict, y, mask)
        return {'loss': loss, 'id': x['id'], 'seq_len': x['seq_len'], 'predict': predict}

    def validation_epoch_end(self, outputs: Union[List[AnyDict], List[List[AnyDict]]]):
        epoch = self.current_epoch + (0 if self.trainer.sanity_checking else 1)
        val_output = outputs[0] if self.test_when_val else outputs
        val_result = self.metric[0].compute()
        val_result['loss'] = sum(batch['loss'] for batch in val_output) / len(val_output)
        self.log_dict({'val/' + k: v for k, v in val_result.items()})
        self.print(f'[{epoch}]\tVAL\t' + '\t'.join(f'{k}={v:.4f}' for k, v in val_result.items()))

        if self.test_when_val:
            test_result = self.metric[1].compute()
            test_result['loss'] = sum(item['loss'] for item in outputs[1]) / len(outputs[1])
            # print(sum(item['loss'] for item in outputs[1]), len(outputs[1]))
            self.log_dict({'test/' + k: v for k, v in test_result.items()})
            # log.debug(f'This test has {self.metric2.n} sents and {getattr(self.metric2, "total", "UNK")} tokens.')
            self.print(f'[{epoch}]\tTEST\t' + '\t'.join(f'{k}={v:.4f}' for k, v in test_result.items()))

        self._val_outputs = [outputs] if not self.test_when_val else outputs

        # if self.trainer.current_epoch + 1 >= 100 and (self.trainer.current_epoch + 1) % 50 == 0:
        #     self.trainer.save_checkpoint(f'{self.trainer.current_epoch}.ckpt')

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        test_result = self.metric[0].compute()
        test_result['loss'] = sum(item['loss'] for item in outputs) / len(outputs)
        if self.logger is not None:
            self.logger.log_metrics(apply_to_dict(test_result, lambda x: x.item()), 0)
        self.log_dict(test_result)
        self._test_outputs = [outputs]

    def on_predict_epoch_start(self) -> None:
        self.predict_start_time = time.time()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]) -> Any:
        x, y = batch
        vp = self.model.set_varpool(VarPool(**x))
        score = self.model(x, vp)
        predict = self.model.decode(score, vp)
        return {'predict': predict}

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        self._predict_time = time.time() - self.predict_start_time
        # self._predict_outputs = [outputs]

    def configure_optimizers(self):
        optimizer_cfg = src.g_cfg.optimizer
        if optimizer_cfg.groups is None or len(optimizer_cfg.groups) == 0:
            params = self.model.parameters()
        else:
            params = [[] for _ in optimizer_cfg.groups]
            default_group = []
            for name, p in self.model.named_parameters():
                matches = [i for i, g in enumerate(optimizer_cfg.groups) if re.match(g.pattern, name)]
                if len(matches) > 1:
                    log.warning(f'{name} is ambiguous: {[optimizer_cfg.groups[m].pattern for m in matches]}')
                if len(matches) > 0:
                    log.debug(f'{name} match {optimizer_cfg.groups[matches[0]].pattern}.')
                    params[matches[0]].append(p)
                else:
                    log.debug(f'{name} match defaults.')
                    default_group.append(p)
            for i in range(len(params)):
                if len(params[i]) == 0:
                    log.warning(f'Nothing matches {optimizer_cfg.groups[i].pattern}')
            params = [{'params': p, **optimizer_cfg.groups[i]} for i, p in enumerate(params) if len(p) > 0]
            params.append({'params': default_group})

        optimizer = instantiate(optimizer_cfg.args, params=params, _convert_='all')

        if (scheduler_cfg := src.g_cfg.scheduler) is None:
            return optimizer

        if scheduler_cfg.get('init_when_running'):
            self._init_scheduler_when_running = scheduler_cfg
            scheduler_cfg = deepcopy(scheduler_cfg)  # create a fake scheduler to make lr_monitor work
            for key in scheduler_cfg.args:
                if isinstance(scheduler_cfg.args[key], str) and scheduler_cfg.args[key].endswith(' epoch'):
                    scheduler_cfg.args[key] = 1

        scheduler = instantiate_no_recursive(scheduler_cfg.args, optimizer=optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': scheduler_cfg.interval,
                'frequency': scheduler_cfg.frequency,
                'monitor': src.g_cfg.watch_field,
                'strict': True
            }
        }

    def write_prediction(self, filename, mode, output=None):
        if output is None:
            output = self._val_outputs[0] if mode == 'val' else self._test_outputs[0]
        output = self.model.preprocess_write(output)
        
        if dist.is_initialized() and (ws := dist.get_world_size()) > 1:
            holder = [None] * ws
            dist.all_gather_object(holder, output)
            if dist.get_rank() > 0:
                return
            else:
                output = reduce(merge_outputs, holder)
    
        ds = self.dm.datasets[mode]
        with open(filename, 'w') as f:
            self.model.write_prediction(f, output, ds, self.dm.vocabs)

    def init_alpha_scheduler(self, command):
        return get_coeff_iter(command, lambda: self.current_epoch)

    def reduce_loss(self, loss, vp):
        return reduce_loss('sum', loss, vp.num_token, vp.batch_size)

    def get_progress_bar_dict(self):
        # don't show the version number, because we use hydra to manage workdir.
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items

    def load_model_inplace(self, path):
        checkpoint = pl_load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'], strict=False)

