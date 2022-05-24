from __future__ import annotations

import torch.nn as nn
from src.my_typing import *


class EncoderBase(nn.Module):
    def __init__(self, embedding: Embedding):
        super().__init__()
        self.bounded_embedding: Embedding = None
        self.bounded_model: BasicModel = None
        self.__dict__['bounded_embedding'] = embedding

    def get_dim(self, field):
        raise NotImplementedError(f'Unrecognized {field=}')

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
