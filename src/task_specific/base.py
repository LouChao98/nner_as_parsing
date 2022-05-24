from __future__ import annotations

import torch
import logging
from collections import defaultdict
import torch.nn as nn
from src.my_typing import *
from src.encoders import MuxEncoder
from src.utils.cl import defaultlist
from src.utils.config import Config
from src.utils.fn import get_coeff_iter
import functools
from fastNLP import seq_len_to_mask

log = logging.getLogger('task')


class TaskBase(nn.Module):
    _function_group = {}

    def __init__(self, encoder: EncoderBase, **cfg):
        # You should not do self.encoder=encoder
        super().__init__()
        self.cfg = Config()
        self.bounded_encoder: EncoderBase = None
        self.bounded_embedding: Embedding = None
        self.bounded_model: BasicModel = None
        self._dynamic_coeff = {}
        self._dynamic_epoch = -1
        self.__dict__['bounded_encoder'] = encoder
        self.__dict__['bounded_embedding'] = encoder.bounded_embedding



    def forward(self, x: TensorDict, vp: VarPool) -> TensorDict:
        raise NotImplementedError

    def loss(self, x: TensorDict, gold: InputDict, vp: VarPool) -> Tuple[Tensor, TensorDict]:
        raise NotImplementedError

    def label_loss(self, logit, target):
        pass

    def decode(self, x: TensorDict, vp: VarPool) -> TensorDict:
        raise NotImplementedError

    def preprocess_write(self, output: List[Dict[str, Any]]):
        batch_size = len(output[0]['id'])  # check one batch
        safe_to_sort = all((len(p) == batch_size) for p in output[0]['predict'].values())

        if safe_to_sort:
            # I will put all predicts in the order of idx, but you have to remove padding by yourself.
            sorted_predicts = defaultdict(defaultlist)
            for batch in output:
                id_, predict = batch['id'], batch['predict']
                for key, value in predict.items():
                    if isinstance(value, Tensor):
                        value = value.detach().cpu().numpy()
                    for one_id, one_value in zip(id_, value):
                        sorted_predicts[key][one_id] = one_value
            unwraped = {}
            for key, value in sorted_predicts.items():
                unwraped[key] = list(value)
            return unwraped
        else:
            raise NotImplementedError('Can not preprocess automatically.')

    def write_prediction(self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]):
        label_vocab = vocabs['span_label']
        for i, length in enumerate(dataset['seq_len'].content):
            word, chart = dataset[i]['raw_words'], predicts['span_headed'][i]
            assert len(word) == length - 2
            s.write(' '.join(word))
            s.write('\n')
            s.write('|'.join([f'{l},{r}({h}) {label_vocab.to_word(label)}' for l, r, h, label in chart if label != 0]))
            # s.write('|'.join([f'{l},{r}({h}) {label_vocab.to_word(label)}' for l, r, h, label in chart if label != 0]))
            s.write('\n')
            s.write(' '.join(map(str, predicts['arc'][i][1:length - 1])))
            s.write('\n\n')
        return s

    def set_varpool(self, vp: VarPool) -> VarPool:
        def _to_mask_label(seq_len, max_len):
            # return: [batch, inclusive left, exclusive right]
            m: Tensor = seq_len_to_mask(seq_len - 1, max_len - 1)
            m = m.unsqueeze(-1) * m.unsqueeze(-2)
            m *= torch.triu(torch.ones(max_len - 1, max_len - 1, device=m.device, dtype=torch.bool), 1)
            return m

        def _to_valid_prediction(seq_len, max_len):
            return seq_len_to_mask(seq_len - 1, max_len - 1)

        vp.add_lazy('mask', 'mask_dep', lambda x: x[:, :-1])
        vp.add_lazy(['seq_len', 'max_len'], 'mask_label', _to_mask_label)
        vp.add_lazy(['seq_len', 'max_len'], 'valid_predict', _to_valid_prediction)  # mask for prediction
        return vp

    def normalize_potential(self, score, fields=None):
        # fields = None to normalize all
        if not self.cfg.potential_normalize:
            return score
        fields = fields if fields is not None else ('arc', 'span')
        var = getattr(self.cfg, 'potential_normalize_var', 1.)
        out = {}
        for field, log_potentials in score.items():
            if field in fields:
                # TODO should I apply mask?
                lp_std, lp_mean = torch.std_mean(log_potentials, list(range(1, log_potentials.ndim)), keepdim=True)
                log_potentials = log_potentials - lp_mean
                log_potentials = log_potentials / ((lp_std + 1e-9) / var)
            out[field] = log_potentials
        return out

    def apply_fencepost(self, x, name):
        if self.cfg.fencepost_mode == 'transformer':
            half_hidden = x.shape[-1] // 2
            x = torch.cat([x[..., 0::2], x[..., 1::2]], -1)
            x = torch.cat([x[:, :-1, :half_hidden], x[:, 1:, half_hidden:]], dim=-1)
        elif self.cfg.fencepost_mode == 'lstm':
            if isinstance(self.bounded_encoder, MuxEncoder):
                sep = []
                for dim in self.bounded_encoder.detailed_dims[name]:
                    half = dim // 2
                    sep.extend((half, half))
                x_sep = x.split(sep, -1)
                x_f, x_b = torch.cat(x_sep[::2], -1), torch.cat(x_sep[1::2], -1)
            else:
                x_f, x_b = x.chunk(2, -1)
            # TODO this is only for sep.
            x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        elif self.cfg.fencepost_mode == 'none':
            pass
        else:
            raise ValueError
        return x

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
        return self

    def set_model(self, model: BasicModel):
        self.__dict__['bounded_model'] = model
        return self

    def set_dynamic_coeff(self, name: str, command: List[str]):
        assert name not in self._dynamic_coeff
        self._dynamic_coeff[name] = get_coeff_iter(command, idx_getter=lambda: self.bounded_model.trainer.current_epoch)

    def get_dynamic_coeff(self):
        if self.bounded_model.trainer.current_epoch != self._dynamic_epoch:
            self._dynamic_epoch = self.bounded_model.trainer.current_epoch
            return {key: next(value) for key, value in self._dynamic_coeff.items()}
        return None

    @classmethod
    def add_impl_to_group(cls, group, spec, pre_hook=None):
        def decorator(func):
            if group not in cls._function_group:
                cls._function_group[group] = {}
            assert spec not in cls._function_group[group]
            log.debug(f"Registering {spec} to group {group} for class {cls}")
            cls._function_group[group][spec] = (func, pre_hook)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def set_impl_in_group(self, group, spec):
        try:
            impl, pre_hook = self._function_group[group][spec]
        except Exception as e:
            log.warn(f"Failed to load {group}: {spec}")
            raise e
        if pre_hook is not None:
            getattr(self, pre_hook)()
        setattr(self, group, functools.partial(impl, self))


class BratBase(TaskBase):
    def __init__(self, encoder: EncoderBase, **cfg):
        super().__init__(encoder, **cfg)

    def normalize_potential(self, score, fields=None):
        # fields = None to normalize all
        if not self.cfg.potential_normalize:
            return score
        fields = fields if fields is not None else ('arc', 'span')
        var = getattr(self.cfg, 'potential_normalize_var', 1.)
        out = {}
        for field, log_potentials in score.items():
            if field in fields:
                # TODO should I apply mask?
                lp_std, lp_mean = torch.std_mean(log_potentials, list(range(1, log_potentials.ndim)), keepdim=True)
                log_potentials = log_potentials - lp_mean
                log_potentials = log_potentials / ((lp_std + 1e-6) / var)
            out[field] = log_potentials
        return out

    def apply_fencepost(self, x, name):
        if self.cfg.fencepost_mode == 'transformer':
            half_hidden = x.shape[-1] // 2
            x = torch.cat([x[..., 0::2], x[..., 1::2]], -1)
            x = torch.cat([x[:, :-1, :half_hidden], x[:, 1:, half_hidden:]], dim=-1)
        elif self.cfg.fencepost_mode == 'lstm':
            if isinstance(self.bounded_encoder, MuxEncoder):
                sep = []
                for dim in self.bounded_encoder.detailed_dims[name]:
                    half = dim // 2
                    sep.extend((half, half))
                x_sep = x.split(sep, -1)
                x_f, x_b = torch.cat(x_sep[::2], -1), torch.cat(x_sep[1::2], -1)
            else:
                x_f, x_b = x.chunk(2, -1)
            # TODO this is only for sep.
            x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        elif self.cfg.fencepost_mode == 'none':
            pass
        else:
            raise ValueError
        return x