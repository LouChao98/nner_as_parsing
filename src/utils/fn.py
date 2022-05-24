import logging
import os
from contextlib import contextmanager
from copy import deepcopy

import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from src.my_typing import *


def instantiate_no_recursive(*args, **kwargs):
    return instantiate(*args, **kwargs, _recursive_=False)


def endless_iter(i, shuffle=None, inplace_shuffle=None):
    while True:
        if shuffle is not None:
            i = shuffle(i)
        if inplace_shuffle is not None:
            inplace_shuffle(i)
        for x in i:
            yield x


def pad_batch(xs, pad_value=0., ignore_none=False):
    """pad two already padded."""
    if ignore_none:
        xs = list(filter(lambda x: x is not None, xs))
    batch_size = sum(item.shape[0] for item in xs)
    longest = max(item.shape[1] for item in xs)
    padded = xs[0].new_full((batch_size, longest, *xs[0].shape[2:]), pad_value)
    offset = 0
    for item in xs:
        padded[offset:offset + item.shape[0], :item.shape[1]] = item
        offset += item.shape[0]
    return padded


def accumulate_to_instance(seq_len, *scores):
    # https://stackoverflow.com/questions/55567838/how-to-avoid-split-and-sum-of-pieces-in-pytorch-or-numpy
    ind = torch.arange(len(seq_len), device=seq_len.device).repeat_interleave(seq_len)
    outs = []
    for s in scores:
        o = torch.zeros(len(seq_len), device=s.device)
        o.index_add_(0, ind, s)
        outs.append(o)
    return outs if len(outs) > 1 else outs[0]


def get_coeff_iter(command, idx_getter=None, validator=None):
    # 1. not (list, tuple, ListConfig): constant alpha
    # 2. List[str]: str should be [value]@[epoch]. eg "[0@0, 0.5@100]". Linearly to value at epoch.
    #               the first term must be @0 (from the beginning)
    if not isinstance(command, (list, tuple, ListConfig)):
        # -123456789 is never reached, so it is endless
        assert command != -123456789
        return iter(lambda: command, -123456789)

    if idx_getter is None:
        _i = 0

        def auto_inc():
            nonlocal _i
            i, _i = _i, _i + 1
            return i

        idx_getter = auto_inc

    def calculate_alpha(value_and_step):
        prev_v, prev_s = value_and_step[0].split('@')
        prev_v, prev_s = float(prev_v), int(prev_s)
        assert prev_s == 0, 'the first step must be 0'
        idx = idx_getter()
        for i in range(1, len(value_and_step)):
            next_v, next_s = value_and_step[i].split('@')
            next_v, next_s = float(next_v), int(next_s)
            rate = (next_v - prev_v) / (next_s - prev_s)
            while idx <= next_s:
                value = prev_v + rate * (idx - prev_s)
                if validator is not None:
                    assert validator(value), f'Bad value in coeff_iter. Get {value}.'
                yield value
                idx = idx_getter()
            prev_v, prev_s = next_v, next_s
        while True:
            yield prev_v

    return iter(calculate_alpha(command))


def add_file_handler(logger, path, level='INFO'):
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(path) == h.baseFilename:
                # file path already added
                return

    file_handler = logging.FileHandler(path, mode='a')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(module)s - [%(levelname)s] - %(message)s',
                                       datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def apply_to_dict(d, func, key_func=None):
    if key_func is not None:
        return {key_func(key): func(value) for key, value in d.items()}
    return {key: func(value) for key, value in d.items()}


def reduce_loss(mode, loss, num_token, num_sentence) -> Tensor:
    if not isinstance(loss, list):
        loss, num_token, num_sentence = [loss], [num_token], [num_sentence]
    assert len(loss) >= 1, 'Nothing to reduce. You should handle this error outside this function.'
    if mode == 'token':
        # average over tokens in a batch
        return sum(loss) / (sum(num_token) + 1e-12)
    elif mode == 'sentence':
        # first average over tokens in a sentence.
        # then average sentences over a batch
        # return sum((l / s).sum() for l, s in zip(loss, seq_len)) / (sum(len(s) for s in seq_len))
        raise NotImplementedError('Deprecated')
    elif mode == 'batch':
        # average over sentences in a batch
        return sum(loss) / (sum(num_sentence) + 1e-12)
    elif mode == 'sum':
        return sum(loss)
    raise ValueError


def instantiate_trainer(callbacks=None, **kwargs):
    if callbacks is not None:
        callbacks = list(callbacks.values())
    return Trainer(callbacks=callbacks, **kwargs)


@contextmanager
def set_cfg(cfg, **kwargs):
    cfg = deepcopy(cfg)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    yield cfg


def split_list(raw, size):
    out = []
    offset = 0
    for s in size:
        out.append(raw[offset: offset + s])
        offset += s
    assert offset == len(raw)
    return out


def merge_outputs(a, b):
    assert a.keys() == b.keys()
    for key in a:
        adata, bdata = a[key], b[key]
        if len(adata) > len(bdata):
            bdata.extend([None] * (len(adata) - len(bdata)))
        else:
            adata.extend([None] * (len(bdata) - len(adata)))
        a[key] = [ai if ai is not None else bi for ai, bi in zip(a[key], b[key])]
    return a