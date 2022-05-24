import logging
import math
from functools import partial

import torch
import torch.distributed as dist
from fastNLP import RandomSampler, SequentialSampler

log = logging.getLogger('Sampler')


class ConstantTokenNumSampler:
    def __init__(self,
                 seq_len,
                 max_token=4096,
                 max_sentence=-1,
                 num_bucket=-1,
                 single_sent_threshold=-1,
                 sort_in_batch=False,
                 shuffle=True,
                 no_drop=False,
                 fully_shuffle=False,
                 force_same_len=False,
                 weight=None):
        """

        :param List[int] seq_len: list[int], 是每个sample的长度。一般可以通过dataset.get_field('seq_len').content传入
        :param int max_token: 每个batch的最大的token数量
        :param int single_sent_threshold: 长度大于阈值的句子强制batch size=1
        :param int max_sentence: 每个batch最多多少个instance, -1表示根据max_token决定
        :param int num_bucket: 将数据按长度拆分为num_bucket个bucket，batch中的sample尽量在bucket之中进行组合，这样可以减少padding。
        :param bool shuffle: shuffle data each epoch. the order is not kept even shuffle=False due to bucket.
        :param bool sort_in_batch: 使得一个batch内sentence长度降序.
        """

        assert len(seq_len) >= num_bucket, 'The number of samples should be larger than buckets.'
        assert num_bucket > 1, 'Use RandomSampler if you do not need bucket.'

        self.seq_len = seq_len
        self.max_token = max_token
        self.max_sentence = max_sentence if max_sentence > 0 else 10000000000000000
        self.single_sent_threshold = single_sent_threshold
        self.sort_in_batch = sort_in_batch
        self.shuffle = shuffle
        self.fully_shuffle = fully_shuffle
        self.epoch = 0

        if force_same_len:
            if weight is None:
                seq_len_indice = [(length, i) for i, length in enumerate(seq_len)]
                seq_len_indice.sort(key=lambda x: x[0])
            else:
                seq_len_indice = [(length, i) for i, length in enumerate(seq_len) for _ in range(weight[i])]
                seq_len_indice.sort(key=lambda x: x[0])
            self.sizes = list(set(seq_len))
            len2idx = dict((l, i) for i, l in enumerate(self.sizes))
            self.buckets = [[] for _ in range(len(self.sizes))]
            for i, l in enumerate(seq_len_indice):
                self.buckets[len2idx[l]].append(i)
        else:
            self.sizes, self.buckets = self.kmeans(seq_len, num_bucket)
            if weight is not None:
                for bucket in self.buckets:
                    for i in range(len(bucket)):
                        bucket[i: i+1] *= weight[bucket[i]]
        self.chunks = [
            min(
                len(bucket),
                max(1, round(size * len(bucket) / max_token),
                    ((len(bucket) + max_sentence - 1) // max_sentence) if max_sentence >= 1 else 0))
            for size, bucket in zip(self.sizes, self.buckets)
        ]

        self.dist = dist.is_initialized()
        self.rank = dist.get_rank() if self.dist else 0
        self.replicas = dist.get_world_size() if self.dist else 1
        self.force_even = self.dist and no_drop
        self.samples = sum(self.chunks) // self.replicas \
                       + (self.replicas * int(sum(self.chunks) % self.replicas) if self.force_even else 0)

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
            g = torch.Generator()
            g.manual_seed(self.epoch)
            range_fn = partial(torch.randperm, generator=g)
        else:
            range_fn = torch.arange
        if self.force_even:
            all = []
            for i in range_fn(len(self.buckets)).tolist():
                split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
                for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                    all.append([self.buckets[i][j] for j in batch.tolist()])

            if self.fully_shuffle:
                all = [all[i] for i in range_fn(len(all))]

            if len(all) % self.replicas != 0:
                for batch in all:
                    if len(batch) < self.replicas:
                        continue
                    all.remove(batch)
                    l = self.replicas - ((len(all) + 1) % self.replicas) + 1
                    for i in range(l):
                        if i < l - 1:
                            all.append([batch[i]])
                        else:
                            all.append(batch[l - 1:])
                    break
            assert len(all) % self.replicas == 0, 'Failed to balance data.'
            for i, batch in enumerate(all):
                if i % self.replicas == self.rank:
                    yield from self._sort_in_batch(batch)
        else:
            all, total, count = [], 0, 0
            for i in range_fn(len(self.buckets)).tolist():
                split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
                for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                    if count == self.samples: break
                    if total % self.replicas == self.rank:
                        count += 1
                        all.append([self.buckets[i][j] for j in batch.tolist()])
                    total += 1

            if self.fully_shuffle:
                all = [all[i] for i in range_fn(len(all))]

            for batch in all:
                yield from self._sort_in_batch(batch)

    def __len__(self):
        return self.samples

    def _sort_in_batch(self, batch):
        singles = []
        if self.single_sent_threshold != -1:
            new_batch = []
            for inst_idx in batch:
                if self.seq_len[inst_idx] >= self.single_sent_threshold:
                    singles.append([inst_idx])
                else:
                    new_batch.append(inst_idx)
            batch = new_batch
        if self.sort_in_batch:
            batch.sort(key=lambda i: -self.seq_len[i])
        if len(batch):
            return [batch] + singles
        else:
            return singles

    def set_epoch(self, epoch: int):
        # This is not a subclass of DistributedSampler
        # this function will never be called by pytorch-lightning.
        self.epoch = epoch

    @staticmethod
    def kmeans(x, k, max_it=32):
        """From https://github.com/yzhangcs/parser/blob/main/supar/utils/alg.py#L7"""

        # the number of clusters must not be greater than the number of datapoints
        x, k = torch.tensor(x, dtype=torch.float), min(len(x), k)
        # collect unique datapoints
        d = x.unique()
        # initialize k centroids randomly
        c = d[torch.randperm(len(d))[:k]]
        # assign each datapoint to the cluster with the closest centroid
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)

        for _ in range(max_it):
            # if an empty cluster is encountered,
            # choose the farthest datapoint from the biggest cluster and move that the empty one
            mask = torch.arange(k).unsqueeze(-1).eq(y)
            none = torch.where(~mask.any(-1))[0].tolist()
            while len(none) > 0:
                for i in none:
                    # the biggest cluster
                    b = torch.where(mask[mask.sum(-1).argmax()])[0]
                    # the datapoint farthest from the centroid of cluster b
                    f = dists[b].argmax()
                    # update the assigned cluster of f
                    y[b[f]] = i
                    # re-calculate the mask
                    mask = torch.arange(k).unsqueeze(-1).eq(y)
                none = torch.where(~mask.any(-1))[0].tolist()
            # update the centroids
            c, old = (x * mask).sum(-1) / mask.sum(-1), c
            # re-assign all datapoints to clusters
            dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)
            # stop iteration early if the centroids converge
            if c.equal(old):
                break
        # assign all datapoints to the new-generated clusters
        # the empty ones are discarded
        assigned = y.unique().tolist()
        # get the centroids of the assigned clusters
        centroids = c[assigned].tolist()
        # map all values of datapoints to buckets
        clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

        return centroids, clusters


class BasicSampler:
    """RandomSampler and SequentialSampler"""

    def __init__(self, seq_len, batch_size, single_sent_threshold, sort_in_batch=False, shuffle=True):

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.single_sent_threshold = single_sent_threshold
        self.sort_in_batch = sort_in_batch
        self.shuffle = shuffle
        self.epoch = 0

        self._sampler = RandomSampler() if shuffle else SequentialSampler()

        self.dist = dist.is_initialized()
        assert not self.dist
        # self.rank = dist.get_rank() if self.dist else 0
        # self.replicas = dist.get_world_size() if self.dist else 1

    def __iter__(self):
        batch = []
        for i in self._sampler(self.seq_len):
            batch.append(i)
            if len(batch) == self.batch_size:
                yield from self._sort_in_batch(batch)
                batch.clear()
        if batch:
            yield from self._sort_in_batch(batch)

    def __len__(self):
        return math.ceil(len(self.seq_len) / self.batch_size)

    def _sort_in_batch(self, batch):
        singles = []
        if self.single_sent_threshold != -1:
            new_batch = []
            for inst_idx in batch:
                if self.seq_len[inst_idx] >= self.single_sent_threshold:
                    singles.append([inst_idx])
                else:
                    new_batch.append(inst_idx)
            batch = new_batch
        if self.sort_in_batch:
            batch.sort(key=lambda i: -self.seq_len[i])
        return [batch] + singles

    def set_epoch(self, epoch: int):
        # This is not a subclass of DistributedSampler
        # this function will never be called by pytorch-lightning.
        self.epoch = epoch


class IncLenSampler:
    """
    Baby Steps described in
        From Baby Steps to Leapfrog: How “Less is More” in Unsupervised Dependency Parsing
    """

    def __init__(self, seq_len, batch_size, sort_in_batch=False, shuffle=True, patience: int = 3):

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.sort_in_batch = sort_in_batch
        self.shuffle = shuffle
        self.epoch = 0
        self.patience = patience  # incr max_len every "patience" epoches.

        self._sampler = RandomSampler() if shuffle else SequentialSampler()

        self.dist = dist.is_initialized()
        assert not self.dist
        # self.rank = dist.get_rank() if self.dist else 0
        # self.replicas = dist.get_world_size() if self.dist else 1

    def __iter__(self):
        batch = []
        for i in self._sampler(self.seq_len):
            batch.append(i)
            if len(batch) == self.batch_size:
                yield from self._sort_in_batch(batch)
                batch.clear()
        if batch:
            yield from self._sort_in_batch(batch)

    def __len__(self):
        return math.ceil(len(self.seq_len) / self.batch_size)

    def _sort_in_batch(self, batch):
        singles = []
        if self.single_sent_threshold != -1:
            new_batch = []
            for inst_idx in batch:
                if self.seq_len[inst_idx] >= self.single_sent_threshold:
                    singles.append([inst_idx])
                else:
                    new_batch.append(inst_idx)
            batch = new_batch
        if self.sort_in_batch:
            batch.sort(key=lambda i: -self.seq_len[i])
        return [batch] + singles

    def set_epoch(self, epoch: int):
        # This is not a subclass of DistributedSampler
        # this function will never be called by pytorch-lightning.
        self.epoch = epoch
