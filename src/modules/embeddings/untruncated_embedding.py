import os
from collections import defaultdict
from typing import Union

import torch
import torch.nn as nn
from fastNLP import Vocabulary
from fastNLP.embeddings.static_embedding import PRETRAIN_STATIC_FILES, _get_embedding_url, cached_path

from . import log


class UntruncatedEmbedding(nn.Module):
    def __init__(
            self,
            vocab: Vocabulary,
            lower: bool = False,  # For embedding words and vocab
            word_transform: bool = None,  # Only for embedding words
            requires_grad: bool = True,
            model_dir_or_name: Union[str, None] = 'en'):
        super().__init__()

        if word_transform is None and lower:
            word_transform = str.lower
        elif word_transform is not None and lower:
            log.warning('I will do str.lower() first, then do word_transform.')
            _word_transform = word_transform
            word_transform = lambda x: _word_transform(x.lower())

        if model_dir_or_name.lower() in PRETRAIN_STATIC_FILES:
            model_url = _get_embedding_url('static', model_dir_or_name.lower())
            model_path = cached_path(model_url, name='embedding')
        else:
            model_path = os.path.abspath(os.path.expanduser(model_dir_or_name))

        # readin embedding
        word_mapping = {}
        assert vocab.unknown is not None, \
            'You have to define a unk token. It is not neccesary to be in the pretrained embedding.'
        word_mapping[vocab.unknown] = 0
        if vocab.padding:
            word_mapping[vocab.padding] = len(word_mapping)
        for spe in vocab.specials:
            word_mapping[spe] = len(word_mapping)

        with open(model_path, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)

            vec_list = [[0] * dim] * len(word_mapping)

            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    if word_transform is not None:
                        word_transformed = word_transform(word)
                        is_transformed = word != word_transformed
                        word = word_transformed
                    else:
                        is_transformed = False
                    nums = list(map(float, parts[-dim:]))
                    # if word == 'unk': # Reproduce zy. Actually this is a bug.
                    #     vec_list[0] = nums
                    assert len(nums) == dim
                    if word not in word_mapping:
                        word_mapping[word_transformed] = len(word_mapping)
                        vec_list.append(nums)
                    else:
                        # if added a transformed word before, replace it with the original vec.
                        if not is_transformed:
                            vec_list[word_mapping[word_transformed]] = nums
                except Exception as e:
                    log.error(f'Error at line {idx}.')
                    raise e

        word_to_word_mapping = [0] * len(vocab)
        _lowered = defaultdict(int)
        _lowered_word = defaultdict(list)
        for word, index_in_vocab in vocab:
            _orig_word = word
            if lower:
                word = word.lower()
            word_to_word_mapping[index_in_vocab] = word_mapping.get(word, 0)
            if word_to_word_mapping[index_in_vocab] == 0:
                if vocab.word_count[word] >= 2:  # min_freq, TODO clean hard code
                    word_to_word_mapping[index_in_vocab] = word_mapping[word] = len(word_mapping)
                    vec_list.append([0] * dim)
                elif lower:
                    _lowered[word] += vocab.word_count[_orig_word]  # sum all form of words
                    _lowered_word[word].append(_orig_word)
        for word, freq in _lowered.items():
            if freq >= 2:
                w_idx = len(word_mapping)
                word_mapping[word] = w_idx
                vec_list.append([0] * dim)
                for orig_word in _lowered_word[word]:
                    word_to_word_mapping[vocab[orig_word]] = w_idx

        # words_to_words: idx2idx from vocab to extended pretrained embedding
        self.register_buffer('words_to_words', torch.tensor(word_to_word_mapping))
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(vec_list))
        self.embed_size = self.embedding.embedding_dim
        # word_mapping: the word2idx in pretrained embedding
        self.word_mapping = word_mapping
        self.requires_grad = requires_grad

    @property
    def requires_grad(self):
        r"""
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        :return:
        """
        requires_grads = set([param.requires_grad for param in self.parameters()])
        if len(requires_grads) == 1:
            return requires_grads.pop()
        else:
            return None

    @requires_grad.setter
    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, words):
        words = self.words_to_words[words]
        words = self.embedding(words)
        return words
