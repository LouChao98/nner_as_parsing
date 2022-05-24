import torch

from ._fn import diagonal, diagonal_copy_, stripe, stripe_compose


@torch.enable_grad()
def cyk(s_span, lens, decode=False, max_margin=False, mbr=False, raw_decode=False):
    if not s_span.requires_grad:
        s_span.requires_grad_(True)

    viterbi = decode or max_margin or raw_decode

    batch, seq_len = s_span.shape[:2]
    s = s_span.new_zeros(batch, seq_len, seq_len).fill_(-1e9)
    s[:, torch.arange(seq_len - 1), torch.arange(seq_len - 1) + 1] = \
        s_span[:, torch.arange(seq_len - 1), torch.arange(seq_len - 1) + 1]

    for w in range(2, seq_len):
        n = seq_len - w
        left = stripe(s, n, w - 1, (0, 1))
        right = stripe(s, n, w - 1, (1, w), 0)
        if viterbi:
            composed = (left + right).max(2)[0]
        else:
            composed = (left + right).logsumexp(2)
        composed = composed + diagonal(s_span, w)
        diagonal_copy_(s, composed, w)

    logZ = s[torch.arange(batch), 0, lens]

    if not decode and not max_margin:
        return logZ

    logZ.sum().backward()
    if max_margin:
        return s_span.grad

    predicted_span = s_span.grad.nonzero()
    if raw_decode:
        return predicted_span
    predicted_span = predicted_span.tolist()
    spans = [[] for _ in range(batch)]
    for (b, start, end) in predicted_span:
        spans[b].append((start, end))
    return spans


@torch.enable_grad()
def cyk2o(s_span, s_comp, lens, decode=False, max_margin=False, mbr=False):
    if not s_span.requires_grad:
        s_span.requires_grad_(True)

    viterbi = decode or max_margin

    batch, seq_len = s_span.shape[:2]
    s = s_span.new_zeros(batch, seq_len, seq_len).fill_(-1e9)
    s[:, torch.arange(seq_len - 1), torch.arange(seq_len - 1) + 1] = \
        s_span[:, torch.arange(seq_len - 1), torch.arange(seq_len - 1) + 1]

    for w in range(2, seq_len):
        n = seq_len - w
        left = stripe(s, n, w - 1, (0, 1))
        right = stripe(s, n, w - 1, (1, w), 0)
        compose = stripe_compose(s_comp, n, w)
        if viterbi:
            composed = (left + right + compose).max(2)[0]
        else:
            composed = (left + right + compose).logsumexp(2)
        composed = composed + diagonal(s_span, w)
        diagonal_copy_(s, composed, w)

    logZ = s[torch.arange(batch), 0, lens]

    if not decode and not max_margin:
        return logZ

    logZ.sum().backward()
    if max_margin:
        return s_span.grad

    predicted_span = s_span.grad.nonzero().tolist()
    spans = [[] for _ in range(batch)]
    for (b, start, end) in predicted_span:
        spans[b].append((start, end))
    return spans
