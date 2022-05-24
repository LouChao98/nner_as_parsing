from __future__ import annotations

import io
import logging
import os
import pprint
import re
import sys
import warnings
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, ProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.overrides.distributed import LightningDistributedModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from src.my_typing import *
from tqdm import tqdm

from .fn import apply_to_dict

log = logging.getLogger('callback')


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def _info(*args, **kwargs):
    log.info(*args, **kwargs)


rank_zero_warn = rank_zero_only(_warn)
rank_zero_info = rank_zero_only(_info)


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""
    def __init__(self, log: str = 'gradients', log_freq: int = 100):
        self.log_mode = log
        self.log_freq = log_freq

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.is_global_zero:
            logger = WatchModelWithWandb.get_wandb_logger(trainer=trainer)
            if logger is not None and self.log_mode != 'none':
                if isinstance(trainer.model, LightningDistributedModule):
                    model = trainer.model.module
                else:
                    model = trainer.model
                logger.watch(model=model, log=self.log_mode, log_freq=self.log_freq)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            if os.path.exists('config/config.yaml'):
                wandb.save('config/config.yaml')
            if os.path.exists('dev.predict.txt'):
                wandb.save('dev.predict.txt')
            if os.path.exists('test.predict.txt'):
                wandb.save('test.predict.txt')
            wandb.finish()

    @staticmethod
    def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
        if trainer.logger is None:
            return None
        logger = trainer.logger
        if not isinstance(logger, Iterable):
            return logger if isinstance(logger, WandbLogger) else None
        for lg in trainer.logger:
            if isinstance(lg, WandbLogger):
                logger = lg
        return logger


class MyProgressBar(ProgressBar):
    """Only one, short, ascii"""
    def __init__(self, refresh_rate: int, process_position: int):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)

    def init_sanity_tqdm(self) -> tqdm:
        bar = tqdm(desc='Validation sanity check',
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=False,
                   ncols=0,
                   ascii=True,
                   file=sys.stdout)
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(desc='Training',
                   initial=self.train_batch_idx,
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=True,
                   smoothing=0,
                   ncols=0,
                   ascii=True,
                   file=sys.stdout)
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(disable=True)
        return bar

    def init_test_tqdm(self) -> tqdm:
        bar = tqdm(desc='Testing',
                   position=self.process_position,
                   disable=self.is_disabled,
                   leave=True,
                   smoothing=0,
                   ncols=0,
                   ascii=True,
                   file=sys.stdout)
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'[{trainer.current_epoch + 1}] train')

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.main_progress_bar.set_description(f'[{trainer.current_epoch + 1}] val')

    def print(
        self, *args, sep: str = " ", end: str = os.linesep, file: Optional[io.TextIOBase] = None, nolock: bool = False
    ):
        _info(sep.join(map(str, args)))
        # active_progress_bar = None
        #
        # if self.main_progress_bar is not None and not self.main_progress_bar.disable:
        #     active_progress_bar = self.main_progress_bar
        # elif self.val_progress_bar is not None and not self.val_progress_bar.disable:
        #     active_progress_bar = self.val_progress_bar
        # elif self.test_progress_bar is not None and not self.test_progress_bar.disable:
        #     active_progress_bar = self.test_progress_bar
        # elif self.predict_progress_bar is not None and not self.predict_progress_bar.disable:
        #     active_progress_bar = self.predict_progress_bar
        #
        # if active_progress_bar is not None:
        #     s = sep.join(map(str, args))
        #     active_progress_bar.write(s, end=end, file=file, nolock=nolock)


class LearningRateMonitorWithEarlyStop(LearningRateMonitor):
    def __init__(self, logging_interval=None, log_momentum=False, minimum_lr=None):
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum)
        self.minimum_lr = minimum_lr
        self.fully_initialized = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule, unused=None):
        if self.minimum_lr is None:
            return
        if len(self.lr_sch_names) < 1:
            return
        main_lr_name = self.lr_sch_names[0]
        if main_lr_name + '/pg1' in self.lrs:
            main_lr = self.lrs[main_lr_name + '/pg1']
        else:
            main_lr = self.lrs[main_lr_name]
        if len(main_lr) == 0:
            return
        else:
            main_lr = main_lr[-1]
        if main_lr < self.minimum_lr and self.fully_initialized:
            trainer.should_stop = True
        elif main_lr >= self.minimum_lr:
            # skip the increasing stage
            self.fully_initialized = True


class BestWatcherCallback(ModelCheckpoint):
    def __init__(
        self,
        monitor: str,
        mode: str,
        hint: bool = True,
        save=False,
        write: str = 'none',
        report: bool = True,
    ):
        assert save is False or isinstance(save, GenDict)
        assert write in ('none', 'new', 'always')

        super().__init__()
        self.monitor = monitor
        self.hint = hint  # hit in logging (not logger).
        self.save = save  # save model checkpoint. dict or False.
        self.write = write  # write predict.
        self.report = report  # report when end or quit to logger and console.

        self.mode = None
        self.best_model_metric = {}
        self.best_model_score = None
        self.best_model_path = ''
        self._save_function = None
        self.write_function = None
        self._last_global_step = -1

        self.__init_monitor_mode(monitor, mode)
        if self.save:
            self.__init_ckpt_dir(save['dirpath'], save['filename'])

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: BasicRunner):
        if self.save:
            self.__resolve_ckpt_dir(trainer)
            self._save_function = trainer.save_checkpoint
        if self.write != 'none':
            self.write_function = pl_module.write_prediction
        self.report = self.report and trainer.logger is not None

    def check_metric(self, trainer, current):
        if current is None:
            return False

        if not isinstance(current, torch.Tensor):
            rank_zero_warn(
                f'{current} is supposed to be a `torch.Tensor`. Saving checkpoint may not work correctly.'
                f' HINT: check the value of {self.monitor} in your validation loop',
                RuntimeWarning,
            )
            current = torch.tensor(current)

        if current.isnan():
            raise RuntimeError('Moniter is Nan.')

        monitor_op = {'min': torch.lt, 'max': torch.gt}[self.mode]
        is_best = monitor_op(current, self.best_model_score)
        is_best = trainer.training_type_plugin.reduce_boolean_decision(is_best)
        return is_best

    def on_validation_end(self, trainer: Trainer, pl_module):
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if (trainer.fast_dev_run  # disable checkpointing with fast_dev_run
                or trainer.sanity_checking  # don't save anything during sanity check
                or self._last_global_step_saved == global_step  # already saved at the last step
            ):  # noqa
            return

        self._validate_monitor_key(trainer)
        self._last_global_step_saved = global_step

        metric = self._monitor_candidates(trainer)
        current = metric.get(self.monitor)
        is_best = self.check_metric(trainer, current)

        if is_best:
            self.best_model_metric = metric
            self.best_model_score = current

        if self.write == 'always' or (self.write == 'new' and is_best):
            self.write_prediction(pl_module)

        if not is_best:
            return

        if self.hint:
            self.do_hint(epoch)
        if self.save and (epoch >= self.save['start_patience']):
            self.save_checkpoint(current, metric.get('epoch'), metric.get('step'), trainer, pl_module, metric)
        if self.report:
            trainer.logger.log_metrics({f'best/{k[5:]}': v
                                        for k, v in metric.items() if k.startswith('test/')}, global_step)

    # def on_predict_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    #     outputs = pl_module._predict_outputs
    #     self.write_function('predict', outputs[0])

    def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        # this will de called even when KeyboardInterrupt
        if self.report:
            metric = apply_to_dict(self.best_model_metric, lambda x: x.item() if isinstance(x, Tensor) else x)
            rank_zero_info(f'Best: {pprint.pformat(metric)}')

    def do_hint(self, epoch):
        rank_zero_info(f'[{epoch + 1}]\tNew best.')

    def save_checkpoint(self, current: Tensor, epoch: int, step: int, trainer, pl_module, ckpt_name_metrics):
        del_filepath = self.best_model_path
        filepath = self._get_metric_interpolated_filepath_name(ckpt_name_metrics, epoch, step, trainer, del_filepath)

        self.best_model_path = filepath
        self._save_model(filepath, trainer, pl_module)

        if del_filepath != '' and filepath != del_filepath:
            self._del_model(del_filepath)

    def write_prediction(self, pl_module):
        if not hasattr(pl_module, '_val_outputs'):
            raise MisconfigurationException('Can not find _val_outputs.'
                                            'This is required because lightning prevent me from getting outputs.')
        outputs = pl_module._val_outputs
        assert len(outputs) in (1, 2)
        self.write_function('dev.predict.txt', 'dev', outputs[0])
        if len(outputs) == 2:
            self.write_function('test.predict.txt', 'test', outputs[1])

    def __resolve_ckpt_dir(self, trainer):
        """
        Determines model checkpoint save directory at runtime. References attributes from the
        trainer's logger to determine where to save checkpoints.
        The base path for saving weights is set in this priority:

        1.  Checkpoint callback's path (if passed in)
        2.  The default_root_dir from trainer if trainer has no logger
        3.  The weights_save_path from trainer, if user provides it
        4.  User provided weights_saved_path

        The base path gets extended with logger name and version (if these are available)
        and subfolder "checkpoints".
        """
        # Todo: required argument `pl_module` is not used
        if self.dirpath is not None:
            return  # short circuit

        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                # the user has changed weights_save_path, it overrides anything
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (trainer.logger.version
                       if isinstance(trainer.logger.version, str) else f'version_{trainer.logger.version}')

            version, name = trainer.training_type_plugin.broadcast((version, trainer.logger.name))

            ckpt_path = os.path.join(save_dir, str(name), version, 'checkpoints')
        else:
            ckpt_path = os.path.join(trainer.weights_save_path, 'checkpoints')

        self.dirpath = ckpt_path

        if not trainer.fast_dev_run and trainer.is_global_zero:
            self._fs.makedirs(self.dirpath, exist_ok=True)

    def _validate_monitor_key(self, trainer):
        metrics = trainer.logger_connector.callback_metrics

        # validate metric
        if self.monitor is not None and not self._is_valid_monitor_key(metrics):
            m = (f"ModelCheckpoint(monitor='{self.monitor}') not found in the returned metrics:"
                 f' {list(metrics.keys())}. '
                 f"HINT: Did you call self.log('{self.monitor}', value) in the LightningModule?")
            raise MisconfigurationException(m)

    def _get_metric_interpolated_filepath_name(
        self,
        ckpt_name_metrics: Dict[str, Any],
        epoch: int,
        step: int,
        trainer,
        del_filepath: Optional[str] = None,
    ) -> str:
        filepath = self.format_checkpoint_name(epoch, step, ckpt_name_metrics)

        version_cnt = 1
        while self.file_exists(filepath, trainer) and filepath != del_filepath:
            filepath = self.format_checkpoint_name(epoch, step, ckpt_name_metrics, ver=version_cnt)
            version_cnt += 1

        return filepath

    def _monitor_candidates(self, trainer):
        monitor_candidates = deepcopy(trainer.logger_connector.callback_metrics)
        monitor_candidates.update(step=trainer.global_step, epoch=trainer.current_epoch)
        return monitor_candidates

    def __init_ckpt_dir(self, dirpath, filename):

        self._fs = get_filesystem(str(dirpath) if dirpath else '')

        if dirpath and self._fs.protocol == 'file':
            dirpath = os.path.realpath(dirpath)

        self.dirpath: Union[str, None] = dirpath or None
        self.filename = filename or None

    def __init_monitor_mode(self, monitor, mode):
        torch_inf = torch.tensor(np.Inf)
        mode_dict = {
            'min': (torch_inf, 'min'),
            'max': (-torch_inf, 'max'),
        }
        if mode not in mode_dict:
            raise MisconfigurationException(f"`mode` can be auto, {', '.join(mode_dict.keys())}, got {mode}")

        self.best_model_score, self.mode = mode_dict[mode]

    @rank_zero_only
    def _del_model(self, filepath: str):
        if self._fs.exists(filepath):
            self._fs.rm(filepath)
            log.debug(f'Removed checkpoint: {filepath}')

    def _save_model(self, filepath: str, trainer, pl_module):
        # Todo: required argument `pl_module` is not used
        # in debugging, track when we save checkpoints
        trainer.dev_debugger.track_checkpointing_history(filepath)

        # make paths
        if trainer.is_global_zero:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the trainer
        if self.save_function is not None:
            self.save_function(filepath, False)  # save_weight_only=False
        else:
            raise ValueError('.save_function() not set')

    @classmethod
    def _format_checkpoint_name(cls, filename: Optional[str], epoch: int, step: int, metrics: Dict[str, Any]) -> str:
        if not filename:
            filename = '{epoch}-{step}'

        groups = re.findall(r'(\{.*?)[:\}]', filename)
        if len(groups) >= 0:
            metrics.update({'epoch': epoch, 'step': step})
            for group in groups:
                name = group[1:]
                filename = filename.replace(group, name + '{' + name)  # REMOVE =, or a messy when loading.
                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)

        filename = filename.replace('/', '_')
        return filename

    def format_checkpoint_name(self, epoch: int, step: int, metrics: Dict[str, Any], ver: Optional[int] = None) -> str:
        """Generate a filename according to the defined template.

        Example::

            >>> tmpdir = os.path.dirname(__file__)
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}')
            >>> os.path.basename(ckpt.format_checkpoint_name(0, 1, metrics={}))
            'epoch=0.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch:03d}')
            >>> os.path.basename(ckpt.format_checkpoint_name(5, 2, metrics={}))
            'epoch=005.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}-{val_loss:.2f}')
            >>> os.path.basename(ckpt.format_checkpoint_name(2, 3, metrics=dict(val_loss=0.123456)))
            'epoch=2-val_loss=0.12.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{missing:d}')
            >>> os.path.basename(ckpt.format_checkpoint_name(0, 4, metrics={}))
            'missing=0.ckpt'
            >>> ckpt = ModelCheckpoint(filename='{step}')
            >>> os.path.basename(ckpt.format_checkpoint_name(0, 0, {}))
            'step=0.ckpt'

        """
        filename = self._format_checkpoint_name(self.filename, epoch, step, metrics)
        if ver is not None:
            filename = '-'.join((filename, f'v{ver}'))

        ckpt_name = f'{filename}.ckpt'
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    def _is_valid_monitor_key(self, metrics):
        return self.monitor in metrics or len(metrics) == 0

    def file_exists(self, filepath: Union[str, Path], trainer) -> bool:
        """
        Checks if a file exists on rank 0 and broadcasts the result to all other ranks, preventing
        the internal state to diverge between ranks.
        """
        exists = self._fs.exists(filepath)
        return trainer.training_type_plugin.broadcast(exists)
