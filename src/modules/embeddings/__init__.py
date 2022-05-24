import logging

log = logging.getLogger('embedding')

from .embedding import Embedding
from .fastnlp_adaptor import FastNLPCharEmbeddingAdaptor, FastNLPEmbeddingAdaptor
from .flair_adaptor import FlairEmbeddingAdaptor
from .partial_trainable_static_embedding import PartialTrainableEmbedding, PartialTrainableEmbeddingAdaptor
from .torch_adaptor import TorchEmbeddingAdaptor
from .transformers_embedding import TransformersAdaptor, TransformersEmbedding
