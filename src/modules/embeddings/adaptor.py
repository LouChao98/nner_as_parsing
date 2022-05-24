
import torch
import torch.nn as nn
from src.my_typing import *

from . import log


class EmbeddingAdaptor(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.register_buffer('device_indicator', torch.zeros(1))

        self._normalize_warned = False

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def device(self):
        return self.device_indicator.device

    @classmethod
    def trigger(cls, target_string: str):
        return False

    def process(self, vocabs, datasets):
        return

    def forward(self, inputs: List[Any]):
        raise NotImplementedError

    def normalize(self, method: str):
        if not self._normalize_warned:
            log.warning(f"{type(self)} didn't implement normalize.")
            self._normalize_warned = True

    def _normalize(self, data: Tensor, method: str):
        with torch.no_grad():
            if method == 'mean+std':
                std, mean = torch.std_mean(data, dim=0, keepdim=True)
                data.sub_(mean).divide_(std)
            elif method == 'mean':
                mean = torch.mean(data, dim=0, keepdim=True)
                data.sub_(mean)
            elif method == 'std':
                std = torch.std(data, dim=0, keepdim=True)
                data.divide_(std)
            else:
                raise ValueError(f'Unrecognized normalize method: {method}')

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    @classmethod
    def apply_adaptor(cls, target_string: str, emb) -> 'EmbeddingAdaptor':
        def all_subclasses(_cls):
            return set(_cls.__subclasses__()).union([s for c in _cls.__subclasses__() for s in all_subclasses(c)])

        candidate = None
        for adaptor in all_subclasses(cls):
            if adaptor.trigger(target_string):
                if candidate is None or issubclass(adaptor, candidate):
                    candidate = adaptor
                elif issubclass(candidate, adaptor):
                    pass
                else:
                    raise KeyError(f'Ambiguous when matching adapters. {candidate} <- {adaptor}')
        assert candidate is not None, f'No available adaptors for {target_string}.'
        log.debug(f'{target_string} choose {candidate}.')
        return candidate(emb)
