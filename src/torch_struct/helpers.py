import math
from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.autograd import Function

from .semirings import LogSemiring, Semiring


class Get(Function):
    @staticmethod
    def forward(ctx, chart, grad_chart, indices):
        ctx.save_for_backward(grad_chart)
        out = chart[indices]
        ctx.indices = indices
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (grad_chart, ) = ctx.saved_tensors
        grad_chart[ctx.indices] += grad_output
        return grad_chart, None, None


class Set(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chart, indices, vals):
        chart[indices] = vals
        ctx.indices = indices
        return chart

    @staticmethod
    def backward(ctx, grad_output):
        z = grad_output[ctx.indices]
        return None, None, z


class Chart:
    def __init__(self, size, potentials, semiring: Semiring, cache=True):
        self.data = semiring.zero_(
            torch.empty(*((semiring.size(), ) + size), dtype=potentials.dtype, device=potentials.device))
        self.grad = self.data.detach().clone().fill_(0.0)
        self.cache = cache
        self.semiring = semiring

    def __getitem__(self, ind):
        I = slice(None)
        if self.cache:
            return Get.apply(self.data, self.grad, (I, I) + ind)
        else:
            return self.data[(I, I) + ind]

    def __setitem__(self, ind, new):
        I = slice(None)
        if self.cache:
            self.data = Set.apply(self.data, (I, I) + ind, new)
        else:
            self.data[(I, I) + ind] = new

    def get(self, ind):
        return Get.apply(self.data, self.grad, ind)

    def set(self, ind, new):
        self.data = Set.apply(self.data, ind, new)


class _Struct:
    def __init__(self, semiring: Semiring = LogSemiring):
        self.semiring = semiring

    def score(self, potentials: Tensor, parts: Tensor, batch_dims=(0, )) -> Tensor:
        """gather all score in parts"""
        score = torch.mul(potentials, parts)
        batch = tuple((score.shape[b] for b in batch_dims))
        return self.semiring.prod(score.view(batch + (-1, )))

    def _bin_length(self, length: int) -> Tuple[int, int]:
        log_N = int(math.ceil(math.log(length, 2)))
        bin_N = int(math.pow(2, log_N))
        return log_N, bin_N

    def _get_dimension_and_requires_grad(self, edge: Union[List[Tensor], Tensor]) -> Tuple[int, ...]:
        if isinstance(edge, (list, tuple)):
            for t in edge:
                t.requires_grad_(True)
            return edge[0].shape
        else:
            edge.requires_grad_(True)
            return edge.shape

    def _chart(self, size, potentials, force_grad):
        return self._make_chart(1, size, potentials, force_grad)[0]

    def _make_chart(self, N, size, potentials, force_grad=False):
        return [(self.semiring.zero_(
            torch.zeros(*((self.semiring.size(), ) + size), dtype=potentials.dtype,
                        device=potentials.device)).requires_grad_(force_grad and not potentials.requires_grad))
                for _ in range(N)]

    def sum(self, edge, lengths=None, _raw=False):
        """
        Compute the (semiring) sum over all structures model.

        Parameters:
            edge : generic params (see class)
            lengths: None or b long tensor mask

        Returns:
            v: b tensor of total sum
        """

        v = self._dp(edge, lengths)[0]
        if _raw:
            return v
        return self.semiring.unconvert(v)

    def marginals(self, edge, lengths=None, _autograd=True, _raw=False, _combine=False):
        """
        Compute the marginals of a structured model.

        Parameters:
            params : generic params (see class)
            lengths: None or b long tensor mask
        Returns:
            marginals: b x (N-1) x C x C table

        """
        if (_autograd or self.semiring is not LogSemiring or not hasattr(self, '_dp_backward')):
            v, edges, _ = self._dp(edge, lengths=lengths, force_grad=True, cache=not _raw)
            if _raw:
                all_m = []
                for k in range(v.shape[0]):
                    obj = v[k].sum(dim=0)

                    marg = torch.autograd.grad(
                        obj,
                        edges,
                        create_graph=True,
                        only_inputs=True,
                        allow_unused=False,
                    )
                    all_m.append(self.semiring.unconvert(self._arrange_marginals(marg)))
                return torch.stack(all_m, dim=0)
            elif _combine:
                obj = v.sum(dim=0).sum(dim=0)
                marg = torch.autograd.grad(obj, edges, create_graph=True, only_inputs=True, allow_unused=False)
                a_m = self._arrange_marginals(marg)
                return a_m
            else:
                obj = self.semiring.unconvert(v).sum(dim=0)
                marg = torch.autograd.grad(obj, edges, create_graph=True, only_inputs=True, allow_unused=False)
                a_m = self._arrange_marginals(marg)
                return self.semiring.unconvert(a_m)
        else:
            v, _, alpha = self._dp(edge, lengths=lengths, force_grad=True)
            return self._dp_backward(edge, lengths, alpha)

    @staticmethod
    def to_parts(spans, extra, lengths=None):
        return spans

    @staticmethod
    def from_parts(spans):
        return spans, None

    def _arrange_marginals(self, marg):
        return marg[0]

    def _dp(self, scores, lengths=None, force_grad=False, cache=True):
        raise NotImplementedError
