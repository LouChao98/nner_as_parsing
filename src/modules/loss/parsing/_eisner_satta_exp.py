import torch

from torch.utils.checkpoint import checkpoint
from torch.autograd import Function
from src.my_typing import *
from ._fn import diagonal, diagonal_copy_, diagonal_copy_v2, diagonal_v2, stripe_version2, stripe_version5

from ._eisner_satta import eisner_satta

class EisnerSattaKL(Function):
    """Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions, Eq 7
    """
    @staticmethod
    def forward(ctx: Any, const, dep, bias, lens) -> Any:
        const = const.detach()
        dep = dep.detach()
        const_marg1, dep_marg1 = eisner_satta(const, dep, lens, bias_on_need_dad=bias, marginal=True)
        const_marg2, dep_marg2 = eisner_satta(const, dep, lens, marginal=True)
        ctx.save_for_backward(const_marg1.detach(), dep_marg1.detach(), const_marg2.detach(), dep_marg2.detach())
        return const.new_ones([])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # return gradient of KL(p_with_bias || p) with respect to p
        const_marg1, dep_marg1, const_marg2, dep_marg2 = ctx.saved_tensors
        return const_marg2 - const_marg1, dep_marg2 - dep_marg1, None, None


@torch.jit.script
def _mix_op1(left, right, left_need_dad, right_need_dad):
    left = left + right_need_dad
    right = right + left_need_dad
    headed = torch.stack([left.max(2)[0], right.max(2)[0]])
    return headed.max(0)[0]

# @torch.jit.script
def _mix_op2(headed, dep):

    return (headed + dep).logsumexp(-2)


@torch.enable_grad()
def eisner_satta_marginal_map(const: Tensor, dependency: Tensor, lens: Tensor):
    if not const.requires_grad:
        const = const.detach().requires_grad_(True)
    if not dependency.requires_grad:
        dependency = dependency.detach().requires_grad_(True)

    dep = dependency[:, 1:, 1:].contiguous()
    root = dependency[:, 1:, 0].contiguous()

    op1 = _mix_op1
    op2 = _mix_op2  #_max_op2 if viterbi else _sum_op2

    B, N, _ = const.shape
    H = N - 1

    s = const.new_full((B, N, N, H), -1e12)
    s_need_dad = const.new_full((B, N, N, H), -1e12)
    s_indicator = torch.zeros_like(s, requires_grad=True)

    ALL = torch.arange(N - 1)

    s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1] + s_indicator[:, ALL, ALL + 1, ALL]
    s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None] + s_indicator[:, ALL, ALL + 1, ALL, None]

    for w in range(2, N):
        n = N - w
        span_score = diagonal(const, w)
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        left_combined = left + right_need_dad
        right_combined = right + left_need_dad
        with torch.no_grad():
            sumed = torch.logaddexp(left_combined, right_combined).logsumexp(3)  # sum out head
            max_span_deduction = sumed.argmax(2, keepdim=True).unsqueeze(-1)

        # complete
        max_left = left_combined.gather(2, max_span_deduction.expand(-1, -1, -1, left_combined.shape[-1])).squeeze(2)
        max_right = right_combined.gather(2, max_span_deduction.expand(-1, -1, -1, left_combined.shape[-1])).squeeze(2)
        headed = torch.where(max_left > max_right, max_left, max_right)
        headed = headed + span_score.unsqueeze(-1)
        headed = headed + diagonal_v2(s_indicator, w)
        diagonal_copy_v2(s, headed, w)

        if w < N - 1:

            # attach
            u = checkpoint(op2, headed.unsqueeze(-1), stripe_version5(dep, n, w))
            # u = op2( headed.unsqueeze(-1), stripe_version5(dep, n, w))
            # if bias_on_need_dad is not None:
            #     u = u + diagonal(bias_on_need_dad, w).unsqueeze(-1)
            diagonal_copy_(s_need_dad, u, w)

    logZ = s[torch.arange(B), 0, lens] + root

    # when you use checkpoint, you cannot use torch.autograd.grad(logZ.sum(), [dependency, constituency])
    logZ.logsumexp(-1).sum().backward(retain_graph=True)
    return const.grad, dependency.grad, s_indicator.grad


def diagonal_copy_v2_offset(x, y, w, lower=False):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:-1]), x.shape[-1] - 1),
                 stride=new_stride,
                 storage_offset=(w * stride[2] if not lower else w * stride[1]) + 1).copy_(y)


def diagonal_copy_offset(x, y, w):
    # size of x: (batch, N, N, nt)
    # size of y: (batch, N, nt)
    # the function aims to copy y to the diagonal of x (dim1 and dim2) without any copy of tensor.
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], seq_len - w, x.shape[-2], x.shape[-1] - 1),
                 stride=new_stride,
                 storage_offset=w * stride[2] + 1).copy_(y)


def _crossentropy_sum(xs, dim: int):
    part_p = torch.logsumexp(xs[..., 0], dim=dim)
    part_q = torch.logsumexp(xs[..., 1], dim=dim)
    log_sm_p = xs[..., 0] - part_p.unsqueeze(dim)
    log_sm_q = xs[..., 1] - part_q.unsqueeze(dim)
    sm_p = log_sm_p.exp()
    return torch.stack((part_p, part_q, torch.sum(xs[..., 2].mul(sm_p) - log_sm_q.mul(sm_p), dim=dim)), dim=-1)


# @torch.jit.script
def _crossentropy_op1(left, right, left_need_dad, right_need_dad):
    left = left + right_need_dad
    right = right + left_need_dad
    headed = torch.stack([_crossentropy_sum(left, 2), _crossentropy_sum(right, 2)])
    return _crossentropy_sum(headed, 0)


# @torch.jit.script
def _crossentropy_op2(a, b):
    return _crossentropy_sum(a + b, 2)


def eisner_satta_crossentropy(const: Tensor, dependency: Tensor, bias: Tensor, lens: Tensor):
    _const = const
    _dep = dependency[:, 1:, 1:]
    _root = dependency[:, 1:, 0]
    dep = torch.zeros(_dep.shape + (3, )).type_as(_dep)
    dep[..., 0] = _dep
    dep[..., 1] = _dep
    root = torch.zeros(_root.shape + (3, )).type_as(_root)
    root[..., 0] = _root
    root[..., 1] = _root
    const = torch.zeros(_const.shape + (3, )).type_as(_const)
    const[..., 0] = _const
    const[..., 1] = _const
    const = _crossentropy_sum(const, -1)

    B, N, *_ = const.shape
    H = N - 1

    s = const.new_full((B, N, N, H, 3), -1e12)
    s_need_dad = const.new_full((B, N, N, H, 3), -1e12)

    ALL = torch.arange(N - 1)
    s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1]
    s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None]

    if bias is not None:
        s_need_dad[:, ALL, ALL + 1, ALL, 0] += diagonal(bias, 1)

    for w in range(2, N):
        n = N - w
        span_score = diagonal(const, w)
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        # complete
        headed = checkpoint(_crossentropy_op1, left.clone(), right.clone(), left_need_dad.clone(), right_need_dad.clone())
        headed = headed + span_score.unsqueeze(-2)
        # if bias is not None:
        #     headed2 = headed.clone()
        #     headed2[..., 0] -= diagonal(bias, w).unsqueeze(-1)
        # else:
        #     headed2 = headed
        diagonal_copy_v2(s, headed, w)

        if w < N - 1:
            # attach
            u = checkpoint(_crossentropy_op2, headed.unsqueeze(-2), stripe_version5(dep, n, w))
            if bias is not None:
                u[..., 0] += diagonal(bias, w).unsqueeze(-1)
            diagonal_copy_(s_need_dad, u, w)

    logZ = s[torch.arange(B), 0, lens] + root
    return _crossentropy_sum(logZ, 1)[:, 2]


def eisner_satta_crossentropy2(const: Tensor, dependency: Tensor, bias: Tensor, lens: Tensor):
    _const = const
    _dep = dependency[:, 1:, 1:]
    _root = dependency[:, 1:, 0]
    dep = torch.zeros(_dep.shape + (3, )).type_as(_dep)
    dep[..., 0] = _dep
    dep[..., 1] = _dep
    root = torch.zeros(_root.shape + (3, )).type_as(_root)
    root[..., 0] = _root
    root[..., 1] = _root
    const = torch.zeros(_const.shape + (3, )).type_as(_const)
    const[..., 0] = _const
    const[..., 1] = _const
    const = _crossentropy_sum(const, -1)

    B, N, *_ = const.shape
    H = N - 1

    s = const.new_full((B, N, N, H, 3), -1e12)
    s_need_dad = const.new_full((B, N, N, H, 3), -1e12)

    ALL = torch.arange(N - 1)
    s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1]
    s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None]

    if bias is not None:
        s[:, ALL, ALL + 1, ALL, 0] -= diagonal(bias, 1)

    for w in range(2, N):
        n = N - w
        span_score = diagonal(const, w)
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        # complete
        headed = checkpoint(_crossentropy_op1, left.clone(), right.clone(), left_need_dad.clone(),
                            right_need_dad.clone())
        headed = headed + span_score.unsqueeze(-2)
        if bias is not None:
            headed2 = headed.clone()
            headed2[..., 0] -= diagonal(bias, w).unsqueeze(-1)
        else:
            headed2 = headed
        diagonal_copy_v2(s, headed2, w)

        if w < N - 1:
            # attach
            u = checkpoint(_crossentropy_op2, headed.unsqueeze(-2), stripe_version5(dep, n, w))
            # if bias is not None:
            #     u[..., 0] += diagonal(bias, w).unsqueeze(-1)
            diagonal_copy_(s_need_dad, u, w)

    logZ = s[torch.arange(B), 0, lens] + root
    return _crossentropy_sum(logZ, 1)[:, 2]