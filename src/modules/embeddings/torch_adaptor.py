from src.my_typing import *
from torch.nn import Embedding as TorchEmbedding

from .adaptor import EmbeddingAdaptor


class TorchEmbeddingAdaptor(EmbeddingAdaptor):
    def __init__(self, emb: TorchEmbedding):
        super().__init__(emb)
        self.emb: TorchEmbedding
        self._embed_size = self.emb.embedding_dim

    @classmethod
    def trigger(cls, target_string):
        return target_string == 'torch.nn.Embedding'

    def forward(self, field: Tensor):
        return self.emb(field)

    def normalize(self, method):
        self._normalize(self.emb.weight.data, method)

