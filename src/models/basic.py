from __future__ import annotations

import logging
import time

import src
import torch
import torch.nn as nn
from hydra.utils import instantiate
from src.modules.embeddings import Embedding
from src.my_typing import *

log = logging.getLogger('model')


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainer: Trainer
        self.embedding: Embedding
        self.encoder: EncoderBase
        self.task: TaskBase
        self.datamodule: DataModule

        self._timing_decoding = False
        self._time = 0

    def setup(self, dm: DataModule):
        self.embedding = Embedding(src.g_cfg.embedding, dm)
        self.encoder = instantiate(src.g_cfg.encoder, embedding=self.embedding)
        self.task = instantiate(src.g_cfg.task, encoder=self.encoder)
        self.datamodule = dm
        self.embedding.__dict__['bounded_model'] = self
        self.encoder.__dict__['bounded_model'] = self
        self.task.__dict__['bounded_model'] = self
        # log.info(self)

    def forward(self, x: InputDict, vp: VarPool, embed=None, encoded=None, return_all=False):
        if embed is None:
            embed = self.embedding(x)
        if encoded is None or embed is None:
            encoded = self.encoder(embed, vp)
        score = self.task.forward(encoded, vp)
        if return_all:
            return embed, encoded, score
        return score

    def loss(self, x: TensorDict, gold: InputDict, vp: VarPool) -> Tuple[Tensor, TensorDict]:
        return self.task.loss(x, gold, vp)

    def decode(self, x: TensorDict, vp: VarPool) -> AnyDict:
        if self._timing_decoding:
            torch.cuda.synchronize(device=None)
            start = time.time()
        result = self.task.decode(x, vp)
        if self._timing_decoding:
            torch.cuda.synchronize(device=None)
            self._time += time.time() - start
        return result

    def confidence(self, x: TensorDict, vp: VarPool, n: int = 1, gold: InputDict = None):
        return self.task.confidence(x, vp, n, gold)

    def normalize_embedding(self, now):
        self.embedding.normalize(now)

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
        self.embedding.set_trainer(trainer)
        self.encoder.set_trainer(trainer)
        self.task.set_trainer(trainer)

    def preprocess_write(self, output: List[Dict[str, Any]]):
        return self.task.preprocess_write(output)

    def write_prediction(self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]) -> IOBase:
        return self.task.write_prediction(s, predicts, dataset, vocabs)

    def set_varpool(self, vp: VarPool) -> VarPool:
        return self.task.set_varpool(vp)
