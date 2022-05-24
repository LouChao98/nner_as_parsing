import torch
from fastNLP.embeddings import StaticEmbedding, TokenEmbedding
from src.my_typing import *

from .adaptor import EmbeddingAdaptor


class FastNLPEmbeddingAdaptor(EmbeddingAdaptor):
    def __init__(self, emb: TokenEmbedding):
        super().__init__(emb)
        self._embed_size = self.emb._embed_size
        self._word_dropout = emb.word_dropout
        self._dropout = emb.dropout_layer.p
        self._normalize_weight = None

    @classmethod
    def trigger(cls, target_string):
        return target_string.startswith('fastNLP')

    def forward(self, field: Tensor):
        return self.emb(field)

    def normalize(self, method):
        emb: torch.nn.Embedding = self.emb.embedding
        if hasattr(self.emb, 'mapped_counts'):
            self.emb: StaticEmbedding
            if self._normalize_weight is None:
                self._normalize_weight = (self.emb.mapped_counts / self.emb.mapped_counts.sum()).unsqueeze(-1)
            mean = (emb.weight.data * self._normalize_weight).sum()
            if method == 'mean':
                emb.weight.data.sub_(mean)
            else:
                std = (((emb.weight.data - mean).pow(2.) * self._normalize_weight).sum() + 1e-6).sqrt()
                if method == 'mean+std':
                    emb.weight.data.sub_(mean)
                emb.weight.data.div_(std)
        else:
            padding_idx = self.emb.get_word_vocab().padding_idx
            if padding_idx == 0:  # no need to use words_to_words
                start_idx = 1
            self._normalize(emb.weight.data[start_idx:], method)


class FastNLPCharEmbeddingAdaptor(FastNLPEmbeddingAdaptor):
    @classmethod
    def trigger(cls, target_string):
        return target_string.startswith('fastNLP') and 'Char' in target_string

    def normalize(self, method):
        emb = self.emb.char_embedding
        start_idx = 1 if self.emb.char_pad_index == 0 else 0
        self._normalize(emb.weight.data[start_idx:], method)


class FastNLPTransformerEmbeddingAdaptor(FastNLPEmbeddingAdaptor):
    # Deprecated
    def __init__(self, emb: TokenEmbedding):
        assert emb.model.include_cls_sep is True
        # emb._word_cls_index = emb.get_word_vocab()['<bos>']
        # emb._word_sep_index = emb.get_word_vocab()['<eos>']

        self.eos = emb.get_word_vocab()['<eos>']
        self.pad = emb.get_word_vocab()['<pad>']
        super().__init__(emb)

    @classmethod
    def trigger(cls, target_string):
        return target_string.startswith('fastNLP') and (('TransformersEmbedding' in target_string)
                                                        or ('RobertaEmbedding' in target_string)
                                                        or ('BertEmbedding' in target_string))

    def forward(self, field: Tensor):
        field = field[:, 1:-1].clone()
        field[field == self.eos] = self.pad
        return self.emb(field)

    def normalize(self, method):
        return
