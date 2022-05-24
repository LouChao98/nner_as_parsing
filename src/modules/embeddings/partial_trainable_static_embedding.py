from __future__ import annotations

import torch
import torch.nn as nn
from fastNLP.embeddings import StaticEmbedding
from src.my_typing import *

from . import log
from .adaptor import EmbeddingAdaptor
from .untruncated_embedding import UntruncatedEmbedding


class PartialTrainableEmbeddingAdaptor(EmbeddingAdaptor):
    def __init__(self, emb: PartialTrainableEmbedding):
        super().__init__(emb)
        self.emb: PartialTrainableEmbedding
        self._embed_size = self.emb.embed_size

    @classmethod
    def trigger(cls, target_string):
        return target_string == 'src.modules.embeddings.PartialTrainableEmbedding'

    def forward(self, field: Tensor):
        return self.emb(field)


class PartialTrainableEmbedding(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 model_dir_or_name: Union[str, None],
                 lower=False,
                 min_freq=1,
                 normalize_method='all',
                 word_transform=None):
        r"""
        :param min_freq: 低于这个频率的词使用 unk+pretrained embedding, 否则使用 emb + pretrained embedding
        -------------------------------
        :param Vocabulary vocab: 词表. StaticEmbedding只会加载包含在词表中的词的词向量，在预训练向量中没找到的使用随机初始化
        :param model_dir_or_name: 可以有两种方式调用预训练好的static embedding：第一种是传入embedding文件夹(文件夹下应该只有一个
            以.txt作为后缀的文件)或文件路径；第二种是传入embedding的名称，第二种情况将自动查看缓存中是否存在该模型，没有的话将自动下载。
            如果输入为None则使用embedding_dim的维度随机初始化一个embedding。
        :param bool lower: 是否将vocab中的词语小写后再和预训练的词表进行匹配。如果你的词表中包含大写的词语，或者就是需要单独
            为大写的词语开辟一个vector表示，则将lower设置为False。
        """
        super().__init__()
        # self.pretrained_embedding = StaticEmbedding(vocab,
        #                                             model_dir_or_name,
        #                                             requires_grad=False,
        #                                             init_method=nn.init.zeros_,
        #                                             lower=lower,
        #                                             min_freq=1,
        #                                             only_use_pretrain_word=True)
        self.pretrained_embedding = UntruncatedEmbedding(vocab, lower, word_transform, False, model_dir_or_name)
        if normalize_method == 'all':
            self.pretrained_embedding.weight.div_(torch.std(self.pretrained_embedding.weight))
        elif normalize_method == 'vector':
            self.pretrained_embedding.weight.div_(torch.std(self.pretrained_embedding.weight, dim=0, keepdim=True))
        elif normalize_method == 'none':
            pass
        else:
            raise NotImplementedError
        self.trainable_embedding = StaticEmbedding(vocab,
                                                   embedding_dim=self.pretrained_embedding.embed_size,
                                                   requires_grad=True,
                                                   init_method=nn.init.zeros_,
                                                   lower=lower,
                                                   min_freq=min_freq)
        self.embed_size = self.pretrained_embedding.embed_size
        log.info(f'{self.pretrained_embedding.embedding.num_embeddings} in pretrained, '
                 f'{self.trainable_embedding.embedding.num_embeddings} in trainable.')

    def forward(self, words):
        return self.trainable_embedding(words) + self.pretrained_embedding(words)
