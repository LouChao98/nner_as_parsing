from __future__ import annotations

from dataclasses import dataclass
from pprint import pformat

import torch
from hydra.utils import instantiate
from src.my_typing import *

from . import log
from .adaptor import EmbeddingAdaptor


@dataclass
class EmbeddingItem:
    name: str
    field: str
    emb: EmbeddingAdaptor


class Embedding(torch.nn.Module):
    """Embedding, plus apply to different fields."""
    def __init__(self, emb_cfg: DictConfig, dm: DataModule):
        super().__init__()
        self.bounded_model: BasicModel = None
        self.emb_cfg = emb_cfg
        vocabs = dm.vocabs
        datasets = dm.datasets

        # process claims
        assert emb_cfg._runtime_normalize in ('none', 'epoch', 'batch')
        self.runtime_normalize = emb_cfg._runtime_normalize
        disabled_fields = []
        if not emb_cfg._use_word:
            disabled_fields.append('words')
        if not emb_cfg._use_subword:
            disabled_fields.append('subwords')
        if not emb_cfg._use_pos:
            disabled_fields.append('pos')
        self.disabled_fields = set(disabled_fields)
        self.cat_output = emb_cfg._cat_output

        # instantiate embeddings
        self.embeds: List[EmbeddingItem] = []
        self.normalize_dict = {'nowhere': [], 'begin': [], 'epoch': [], 'batch': []}

        for name, cfg in self.emb_cfg.items():
            if name.startswith('_'):
                continue
            if cfg is None:
                continue
            if cfg.field in self.disabled_fields:
                continue
            instantiate_args = {}
            if cfg.get('requires_vocab', True):
                instantiate_args['vocab'] = vocabs[cfg.field]
            if cfg.get('normalize_word', False):
                instantiate_args['word_transform'] = dm.normalize_one_word_func
            emb = instantiate(cfg.args, **instantiate_args)
            emb = EmbeddingAdaptor.apply_adaptor(cfg.args._target_, emb)
            emb.process(vocabs, datasets)
            self.add_module(name, emb)
            self.embeds.append(EmbeddingItem(name, cfg.field, emb))

            # configure normalize
            if self.runtime_normalize != 'none':
                when, method = self.runtime_normalize, 'mean+std'
            elif (normalize_cfg := cfg.get('normalize')) is not None:
                when, method = normalize_cfg.when, normalize_cfg.method
                assert method in ('mean+std', 'mean', 'std', None)
            else:
                when, method = 'nowhere', None
            self.normalize_dict[when].append((name, method))

        log.info(f'Emb: {", ".join(e.name for e in self.embeds)}')
        log.info(f'Normalize plan: {pformat({k: v for k, v in self.normalize_dict.items() if len(v)})}')

        self.embed_size = sum(e.embed_size for e in self)
        # if self.cat_output:
        #     self.embed_size = sum(e.embed_size for e in self)
        # else:
        #     self.embed_size = {e.name: e.emb.embed_size for e in self.embeds}

    def forward(self, x):
        emb = [item.emb(x[item.field]) for item in self.embeds]
        if self.cat_output:
            return torch.cat(emb, dim=-1)
        else:
            return emb

    def normalize(self, now):
        # if len(self.normalize_dict[now]):
        #     log.info(f'{now} {self.normalize_dict[now]}')
        for name, method in self.normalize_dict[now]:
            getattr(self, name).normalize(method)

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
        for e in self:
            e.set_trainer(trainer)

    def __getitem__(self, key):
        return self.embeds[key].emb

    def __iter__(self):
        return map(lambda e: e.emb, self.embeds)

    def __len__(self):
        return len(self.embeds)
