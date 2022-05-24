import torch
from torch.utils.checkpoint import checkpoint

from src.my_typing import *
from ._fn import diagonal, diagonal_copy_, diagonal_copy_v2, diagonal_v2, stripe_version2, stripe_version5


@torch.jit.script
def _sum_op1(left, right, left_need_dad, right_need_dad):
    left = left + right_need_dad
    right = right + left_need_dad
    headed = torch.stack([left.logsumexp(2), right.logsumexp(2)])
    return headed.logsumexp(0)


@torch.jit.script
def _max_op1(left, right, left_need_dad, right_need_dad):
    left = left + right_need_dad
    right = right + left_need_dad
    headed = torch.stack([left.max(2)[0], right.max(2)[0]])
    return headed.max(0)[0]


@torch.jit.script
def _sum_op2(a, b):
    return (a + b).logsumexp(-2)


@torch.jit.script
def _max_op2(a, b):
    return (a + b).max(-2)[0]


@torch.jit.script
def _sum_op3(a, b, c):
    return (a + b).logsumexp(-2) + c


@torch.jit.script
def _max_op3(a, b, c):
    return (a + b).max(-2)[0] + c


@torch.enable_grad()
def eisner_satta(const: Tensor, dependency: Tensor, lens: Tensor, decode=False, max_marginal=False,
                 need_span_head=False, bias_on_need_dad=None, marginal=False):
    if not const.requires_grad:
        const = const.detach().requires_grad_(True)
    if not dependency.requires_grad:
        dependency = dependency.detach().requires_grad_(True)

    dep = dependency[:, 1:, 1:].contiguous()
    root = dependency[:, 1:, 0].contiguous()

    viterbi = decode or max_marginal
    op1 = _max_op1 if viterbi else _sum_op1
    op2 = _max_op2 if viterbi else _sum_op2

    B, N, _ = const.shape
    H = N - 1

    s = const.new_full((B, N, N, H), -1e12)
    s_need_dad = const.new_full((B, N, N, H), -1e12)

    if need_span_head:
        s_indicator = torch.zeros_like(s, requires_grad=True)

    ALL = torch.arange(N - 1)

    if need_span_head:
        s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1] + s_indicator[:, ALL, ALL + 1, ALL]
        s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None] + s_indicator[:, ALL, ALL + 1, ALL,
                                                                                   None]
    else:
        s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1]
        s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None]

    # fix
    if bias_on_need_dad is not None:
        if marginal:
            bias_on_need_dad = bias_on_need_dad.detach().requires_grad_()
        s[:, ALL, ALL + 1, ALL] -= diagonal(bias_on_need_dad, 1)

    for w in range(2, N):
        n = N - w
        span_score = diagonal(const, w)
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        # complete
        headed = checkpoint(op1, left.clone(), right.clone(), left_need_dad.clone(), right_need_dad.clone())
        headed = headed + span_score.unsqueeze(-1)
        if need_span_head:
            headed = headed + diagonal_v2(s_indicator, w)
        if bias_on_need_dad is not None:
            headed2 = headed - diagonal(bias_on_need_dad, w).unsqueeze(-1)
        else:
            headed2 = headed
        diagonal_copy_v2(s, headed2, w)

        if w < N - 1:
            # attach
            u = checkpoint(op2, headed.unsqueeze(-1), stripe_version5(dep, n, w))
            # if bias_on_need_dad is not None:
            #     u = u + diagonal(bias_on_need_dad, w).unsqueeze(-1)
            diagonal_copy_(s_need_dad, u, w)

    logZ = s[torch.arange(B), 0, lens] + root

    if not viterbi and not marginal:
        if need_span_head:
            return logZ.logsumexp(-1), s_indicator
        return logZ.logsumexp(-1)

    # when you use checkpoint, you cannot use torch.autograd.grad(logZ.sum(), [dependency, constituency])
    if viterbi:
        logZ.max(-1)[0].sum().backward()
    else:
        logZ.logsumexp(-1).sum().backward()

    if max_marginal or marginal:
        # if bias_on_need_dad is not None:
        #     # breakpoint()
        #     print('>>>', const.grad)
        #     const.grad += bias_on_need_dad.grad
        if need_span_head:
            return const.grad, dependency.grad, s_indicator.grad
        return const.grad, dependency.grad
    spans = [[] for _ in range(B)]
    if need_span_head:
        raw_spans = s_indicator.grad.nonzero().tolist()
        for b, start, end, head in raw_spans:
            spans[b].append((start, end, head))
    else:
        raw_spans = const.grad.nonzero().tolist()
        for b, start, end in raw_spans:
            spans[b].append((start, end))
        for span in spans:
            span.sort(key=lambda x: (x[0], -x[1]))

    arcs = s.new_zeros(B, N, dtype=torch.long)
    raw_arcs = dependency.grad.nonzero()
    arcs[raw_arcs[:, 0], raw_arcs[:, 1]] = raw_arcs[:, 2]

    return spans, arcs


# @torch.jit.script
def _entropy_sum(xs, dim: int):
    part = torch.logsumexp(xs[..., 0], dim=dim)
    log_sm = xs[..., 0] - part.unsqueeze(dim)
    sm = log_sm.exp()
    return torch.stack((part, torch.sum(xs[..., 1].mul(sm) - log_sm.mul(sm), dim=dim)), dim=-1)


# @torch.jit.script
def _entropy_op1(left, right, left_need_dad, right_need_dad):
    left = left + right_need_dad
    right = right + left_need_dad
    headed = torch.stack([_entropy_sum(left, 2), _entropy_sum(right, 2)])
    return _entropy_sum(headed, 0)


# @torch.jit.script
def _entropy_op2(a, b):
    return _entropy_sum(a + b, 2)


def eisner_satta_entropy(const: Tensor, dependency: Tensor, lens: Tensor, fix=False):
    _const = const
    _dep = dependency[:, 1:, 1:]
    _root = dependency[:, 1:, 0]
    dep = torch.zeros(_dep.shape + (2,)).type_as(_dep)
    dep[..., 0] = _dep
    root = torch.zeros(_root.shape + (2,)).type_as(_root)
    root[..., 0] = _root
    if fix:
        const = torch.zeros(_const.shape + (2, )).type_as(_const)
        const[..., 0] = _const
        const = _entropy_sum(const, -1)
    else:
        const = torch.zeros(_const.shape + (2,)).type_as(_const)
        const[..., 0] = _const

    B, N, *_ = const.shape
    H = N - 1

    s = const.new_full((B, N, N, H, 2), -1e12)
    s_need_dad = const.new_full((B, N, N, H, 2), -1e12)

    ALL = torch.arange(N - 1)
    s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1]
    s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None]

    for w in range(2, N):
        n = N - w
        span_score = diagonal(const, w)
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        # complete
        headed = checkpoint(_entropy_op1, left.clone(), right.clone(), left_need_dad.clone(), right_need_dad.clone())
        headed = headed + span_score.unsqueeze(-2)
        diagonal_copy_v2(s, headed, w)

        if w < N - 1:
            # attach
            u = checkpoint(_entropy_op2, headed.unsqueeze(-2), stripe_version5(dep, n, w))
            diagonal_copy_(s_need_dad, u, w)

    logZ = s[torch.arange(B), 0, lens] + root
    return _entropy_sum(logZ, 1)[:, 1]


@torch.enable_grad()
def eisner_satta_v2(dependency: Tensor, span_head_score: Tensor, lens, decode=False, max_margin=False):
    # span_head_score represent a span [i, j) with head word k (ouside the span).
    # k is the index without root (span style).
    B, N = span_head_score.shape[:2]
    H = N - 1

    if decode or max_margin:
        dependency = dependency.detach().clone().requires_grad_(True)
        span_head_score = span_head_score.detach().clone().requires_grad_(True)

    dep = dependency[:, 1:, 1:].contiguous()
    root = dependency[:, 1:, 0].contiguous()
    viterbi = decode or max_margin

    s = span_head_score.new_zeros(B, N, N, H).fill_(-1e9)
    s_need_dad = span_head_score.new_zeros(B, N, N, H).fill_(-1e9)

    ALL = torch.arange(N - 1)
    s[:, ALL, ALL + 1, ALL] = 0
    s_need_dad[:, ALL, ALL + 1, :] = dep[:, ALL] + span_head_score[:, ALL, ALL + 1, :]

    op1 = _max_op1 if viterbi else _sum_op1
    op3 = _max_op3 if viterbi else _sum_op3

    for w in range(2, N):
        n = N - w
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        # complete
        headed = checkpoint(op1, left.clone(), right.clone(), left_need_dad.clone(), right_need_dad.clone())
        diagonal_copy_v2(s, headed, w)

        if w < N - 1:
            # attach, also add "dad generate span"
            u = checkpoint(op3, headed.unsqueeze(-1), stripe_version5(dep, n, w), diagonal(span_head_score, w))
            diagonal_copy_(s_need_dad, u, w)

    logZ = s[torch.arange(B), 0, lens] + root

    if not viterbi:
        return logZ.logsumexp(-1)

    logZ.max(-1)[0].sum().backward()

    if decode:
        predicted_arc = s.new_zeros(B, N).long()
        arc = dependency.grad.nonzero()
        predicted_arc[arc[:, 0], arc[:, 1]] = arc[:, 2]

        span_head_grad = span_head_score.grad.nonzero().tolist()
        spans = [[] for _ in range(span_head_score.shape[0])]
        for batch, left, right, head in span_head_grad:
            spans[batch].append((left, right))
            # spans[batch].append((left, right, head + 1))
        out = (predicted_arc, spans)

    else:
        out = (dependency.grad, span_head_score.grad)
    del s, s_need_dad
    return out


@torch.enable_grad()
def eisner_satta_v3(dependency: Union[Tensor, None], head_span_score: Tensor, lens, decode=False, max_margin=False):
    # span_head_score represent a span [i, j) with head word k generating this span (inside the span).
    # k is the index without root (span style).
    B, N = head_span_score.shape[:2]
    H = N - 1

    if dependency is None:
        dependency = head_span_score.new_zeros(B, N, N)
    if decode or max_margin:
        dependency = dependency.detach().clone().requires_grad_(True)
        head_span_score = head_span_score.detach().clone().requires_grad_(True)

    dep = dependency[:, 1:, 1:].contiguous()
    root = dependency[:, 1:, 0].contiguous()
    viterbi = decode or max_margin

    s = dependency.new_zeros(B, N, N, H).fill_(-1e9)
    s_close = dependency.new_zeros(B, N, N, H).fill_(-1e9)
    s_need_dad = dependency.new_zeros(B, N, N, H).fill_(-1e9)

    ALL = torch.arange(H)
    s[:, ALL, ALL + 1, ALL] = 0
    s_close[:, ALL, ALL + 1, ALL] = 0 if head_span_score is None else head_span_score[:, ALL, ALL + 1, ALL]
    s_need_dad[:, ALL, ALL + 1, :] = dep[:, ALL] + s_close[:, ALL, ALL + 1, ALL].unsqueeze(-1)

    op1 = _max_op1 if viterbi else _sum_op1
    op2 = _max_op2 if viterbi else _sum_op2

    for w in range(2, N):
        n = N - w
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        # complete
        headed = checkpoint(op1, left.clone(), right.clone(), left_need_dad.clone(), right_need_dad.clone())
        diagonal_copy_v2(s, headed, w)
        # complete closed span
        headed = headed + diagonal_v2(head_span_score, w)
        diagonal_copy_v2(s_close, headed, w)

        if w < N - 1:
            # attach
            u = checkpoint(op2, headed.unsqueeze(-1), stripe_version5(dep, n, w))
            diagonal_copy_(s_need_dad, u, w)

    logZ = s_close[torch.arange(B), 0, lens] + root

    if not viterbi:
        return logZ.logsumexp(-1)

    logZ.max(-1)[0].sum().backward()

    if decode:
        predicted_arc = s.new_zeros(B, N).long()
        arc = dependency.grad.nonzero()
        predicted_arc[arc[:, 0], arc[:, 1]] = arc[:, 2]

        span_head_grad = head_span_score.grad.nonzero().tolist()
        spans = [[] for _ in range(head_span_score.shape[0])]
        for batch, left, right, head in span_head_grad:
            spans[batch].append((left, right))
            # spans[batch].append((left, right, head + 1))
        out = (predicted_arc, spans)
    else:
        out = (dependency.grad, head_span_score.grad)
    del s, s_need_dad, s_close
    return out


@torch.enable_grad()
def eisner_satta_for_decode(const, dependency, lens, raw=False, need_span_head=False):
    const = const.requires_grad_(True)
    dependency = dependency.requires_grad_(True)

    dep = dependency[:, 1:, 1:].contiguous()
    root = dependency[:, 1:, 0].contiguous()

    B, N, _ = const.shape
    H = N - 1

    s = const.new_full((B, N, N, H), -1e12)
    s_need_dad = const.new_full((B, N, N, H), -1e12)

    if need_span_head:
        s_indicator = torch.zeros_like(s, requires_grad=True)

    ALL = torch.arange(N - 1)
    if need_span_head:
        s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1] + s_indicator[:, ALL, ALL + 1, ALL]
        s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None] + s_indicator[:, ALL, ALL + 1, ALL,
                                                                                   None]
    else:
        s[:, ALL, ALL + 1, ALL] = const[:, ALL, ALL + 1]
        s_need_dad[:, ALL, ALL + 1] = dep[:, ALL] + const[:, ALL, ALL + 1, None]

    _op2 = _max_op2

    for w in range(2, N):
        n = N - w
        span_score = diagonal(const, w)
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)

        # headed = checkpoint(_max_op1, left.clone(), right.clone(), left_need_dad.clone(), right_need_dad.clone())
        headed = _max_op1(left.clone(), right.clone(), left_need_dad.clone(), right_need_dad.clone())
        headed = headed + span_score.unsqueeze(-1)
        if need_span_head:
            headed = headed + diagonal_v2(s_indicator, w)
        diagonal_copy_v2(s, headed, w)

        if w < N - 1:
            # u = checkpoint(_max_op2, headed.unsqueeze(-1), stripe_version5(dep, n, w))
            u = _op2(headed.unsqueeze(-1), stripe_version5(dep, n, w))
            diagonal_copy_(s_need_dad, u, w)

    logZ = s[torch.arange(B), 0, lens] + root

    # when you use checkpoint, you cannot use torch.autograd.grad(logZ.sum(), [dependency, constituency])
    logZ.max(-1)[0].sum().backward()

    arcs = s.new_zeros(B, N, dtype=torch.long)
    raw_arcs = dependency.grad.nonzero()
    arcs[raw_arcs[:, 0], raw_arcs[:, 1]] = raw_arcs[:, 2]

    if raw:
        raw_spans = const.grad.nonzero()
        return raw_spans, arcs
    # breakpoint()
    spans = [[] for _ in range(B)]
    if need_span_head:
        raw_spans = s_indicator.grad.nonzero().tolist()
        for b, start, end, head in raw_spans:
            spans[b].append((start, end, head))
    else:
        raw_spans = const.grad.nonzero().tolist()
        for b, start, end in raw_spans:
            spans[b].append((start, end))
    for span in spans:
        span.sort(key=lambda x: (x[0], -x[1]))

    return spans, arcs
