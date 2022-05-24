from typing import List, Tuple

import torch


def eisner(scores, seq_len):
    batch_size, max_len, _ = scores.shape
    scores = scores.permute(2, 1, 0).contiguous()
    s_i = torch.full_like(scores, -1e12)
    s_c = torch.full_like(scores, -1e12)
    p_i = scores.new_zeros(max_len, max_len, batch_size).long()
    p_c = scores.new_zeros(max_len, max_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, max_len):
        n = max_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i, r) + C(j, r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j, i) = max(C(i, r) + C(j, r+1) + S(j, i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i, j) = max(C(i, r) + C(j, r+1) + S(i, j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j, i) = max(C(r, i) + I(j, r)), i <= r < j
        cl = stripe(s_c, n, w, dim=0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i, j) = max(I(i, r) + C(r, j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][seq_len.ne(w)] = -1e12
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()

    def backtrack(p_i, p_c, heads, i, j, complete):
        if i == j:
            return
        if complete:
            r = p_c[i, j]
            backtrack(p_i, p_c, heads, i, r, False)
            backtrack(p_i, p_c, heads, r, j, True)
        else:
            r, heads[j] = p_i[i, j], i
            i, j = sorted((i, j))
            backtrack(p_i, p_c, heads, i, r, True)
            backtrack(p_i, p_c, heads, j, r + 1, True)

    for i, length in enumerate(seq_len.tolist()):
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(scores.device))
    return pad(predicts, total_length=max_len)


def eisner2o(scores, seq_len):
    r"""
    Second-order Eisner algorithm for projective decoding.
    This is an extension of the first-order one that further incorporates sibling scores into tree scoring.

    References:
        - Ryan McDonald and Fernando Pereira. 2006.
          `Online Learning of Approximate Dependency Parsing Algorithms`_.

    Args:
        scores (~torch.Tensor, ~torch.Tensor):
            A tuple of two tensors representing the first-order and second-order scores repectively.
            The first (``[batch_size, seq_len, seq_len]``) holds scores of all dependent-head pairs.
            The second (``[batch_size, seq_len, seq_len, seq_len]``) holds scores of all dependent-head-sibling triples.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting projective parse trees.

    Examples:
        >>> s_arc = torch.tensor([[[ -2.8092,  -7.9104,  -0.9414,  -5.4360],
                                   [-10.3494,  -7.9298,  -3.6929,  -7.3985],
                                   [  1.1815,  -3.8291,   2.3166,  -2.7183],
                                   [ -3.9776,  -3.9063,  -1.6762,  -3.1861]]])
        >>> s_sib = torch.tensor([[[[ 0.4719,  0.4154,  1.1333,  0.6946],
                                    [ 1.1252,  1.3043,  2.1128,  1.4621],
                                    [ 0.5974,  0.5635,  1.0115,  0.7550],
                                    [ 1.1174,  1.3794,  2.2567,  1.4043]],
                                   [[-2.1480, -4.1830, -2.5519, -1.8020],
                                    [-1.2496, -1.7859, -0.0665, -0.4938],
                                    [-2.6171, -4.0142, -2.9428, -2.2121],
                                    [-0.5166, -1.0925,  0.5190,  0.1371]],
                                   [[ 0.5827, -1.2499, -0.0648, -0.0497],
                                    [ 1.4695,  0.3522,  1.5614,  1.0236],
                                    [ 0.4647, -0.7996, -0.3801,  0.0046],
                                    [ 1.5611,  0.3875,  1.8285,  1.0766]],
                                   [[-1.3053, -2.9423, -1.5779, -1.2142],
                                    [-0.1908, -0.9699,  0.3085,  0.1061],
                                    [-1.6783, -2.8199, -1.8853, -1.5653],
                                    [ 0.3629, -0.3488,  0.9011,  0.5674]]]])
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> eisner2o((s_arc, s_sib), mask)
        tensor([[0, 2, 0, 2]])

    .. _Online Learning of Approximate Dependency Parsing Algorithms:
        https://www.aclweb.org/anthology/E06-1011/
    """

    # the end position of each sentence in a batch
    s_arc, s_sib = scores
    batch_size, max_len, _ = s_arc.shape
    # [seq_len, seq_len, batch_size]
    s_arc = s_arc.permute(2, 1, 0).contiguous()
    # [seq_len, seq_len, seq_len, batch_size]
    s_sib = s_sib.permute(2, 1, 3, 0).contiguous()
    s_i = torch.full_like(s_arc, -1e12)
    s_s = torch.full_like(s_arc, -1e12)
    s_c = torch.full_like(s_arc, -1e12)
    p_i = s_arc.new_zeros(max_len, max_len, batch_size).long()
    p_s = s_arc.new_zeros(max_len, max_len, batch_size).long()
    p_c = s_arc.new_zeros(max_len, max_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, max_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = max_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # I(j->i) = max(I(j->r) + S(j->r, i)), i < r < j |
        #               C(j->j) + C(i->j-1))
        #           + s(j->i)
        # [n, w, batch_size]
        il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
        il += stripe(s_sib[range(w, n + w), range(n)], n, w, (0, 1))
        # [n, 1, batch_size]
        il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
        # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
        il[:, -1] = il0.index_fill_(0, seq_len.new_tensor(0), 0).squeeze(1)
        il_span, il_path = il.permute(2, 0, 1).max(-1)
        s_i.diagonal(-w).copy_(il_span + s_arc.diagonal(-w))
        p_i.diagonal(-w).copy_(il_path + starts + 1)
        # I(i->j) = max(I(i->r) + S(i->r, j), i < r < j |
        #               C(i->i) + C(j->i+1))
        #           + s(i->j)
        # [n, w, batch_size]
        ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
        ir += stripe(s_sib[range(n), range(w, n + w)], n, w)
        ir[0] = -1e12
        # [n, 1, batch_size]
        ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
        ir[:, 0] = ir0.squeeze(1)
        ir_span, ir_path = ir.permute(2, 0, 1).max(-1)
        s_i.diagonal(w).copy_(ir_span + s_arc.diagonal(w))
        p_i.diagonal(w).copy_(ir_path + starts)

        # [n, w, batch_size]
        slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        slr_span, slr_path = slr.permute(2, 0, 1).max(-1)
        # S(j, i) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(-w).copy_(slr_span)
        p_s.diagonal(-w).copy_(slr_path + starts)
        # S(i, j) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(w).copy_(slr_span)
        p_s.diagonal(w).copy_(slr_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        # disable multi words to modify the root
        s_c[0, w][seq_len.ne(w)] = -1e12
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_s, p_c, heads, i, j, flag):
        if i == j:
            return
        if flag == 'c':
            r = p_c[i, j]
            backtrack(p_i, p_s, p_c, heads, i, r, 'i')
            backtrack(p_i, p_s, p_c, heads, r, j, 'c')
        elif flag == 's':
            r = p_s[i, j]
            i, j = sorted((i, j))
            backtrack(p_i, p_s, p_c, heads, i, r, 'c')
            backtrack(p_i, p_s, p_c, heads, j, r + 1, 'c')
        elif flag == 'i':
            r, heads[j] = p_i[i, j], i
            if r == i:
                r = i + 1 if i < j else i - 1
                backtrack(p_i, p_s, p_c, heads, j, r, 'c')
            else:
                backtrack(p_i, p_s, p_c, heads, i, r, 'i')
                backtrack(p_i, p_s, p_c, heads, r, j, 's')

    preds = []
    p_i = p_i.permute(2, 0, 1).cpu()
    p_s = p_s.permute(2, 0, 1).cpu()
    p_c = p_c.permute(2, 0, 1).cpu()
    for i, length in enumerate(seq_len.tolist()):
        heads = p_c.new_zeros(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_s[i], p_c[i], heads, 0, length, 'c')
        preds.append(heads.to(seq_len.device))

    return pad(preds, total_length=max_len)

def tarjan(sequence):
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices that make up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


def isprojective(sequence):
    r"""
    Checks if a dependency tree is projective.
    This also works for partial annotation.

    Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
    which are hard to detect in the scenario of partial annotation.

    Args:
        sequence (list[int]):
            A list of head indices.

    Returns:
        ``True`` if the tree is projective, ``False`` otherwise.

    Examples:
        >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
        False
        >>> CoNLL.isprojective([3, -1, 2])
        False
    """

    pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i + 1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return False
            if lj <= hi <= rj and hj == di:
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                return False
    return True


def istree(sequence, proj=False, multiroot=False):
    r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

    if proj and not isprojective(sequence):
        return False
    n_roots = sum(head == 0 for head in sequence)
    if n_roots == 0:
        return False
    if not multiroot and n_roots > 1:
        return False
    if any(i == head for i, head in enumerate(sequence, 1)):
        return False
    return next(tarjan(sequence), None) is None


def pad(tensors, padding_value=0, total_length=None):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors) for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def stripe(x, n, w, offset=(0, 0), dim=1):
    r"""Returns a diagonal stripe of the tensor.
    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.
    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    """
    seq_len = x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def find_dep_boundary(heads: List[int], head_in_span) -> List[Tuple[int, int, int]]:
    left_bd = [i for i in range(len(heads))]
    right_bd = [i + 1 for i in range(len(heads))]

    for child_idx, head_idx in enumerate(heads):
        if head_idx > 0:
            if left_bd[child_idx] < left_bd[head_idx - 1]:
                left_bd[head_idx - 1] = left_bd[child_idx]

            elif child_idx > right_bd[head_idx - 1] - 1:
                right_bd[head_idx - 1] = child_idx + 1
                while head_idx != 0:
                    if heads[head_idx - 1] > 0 and child_idx + 1 > right_bd[heads[head_idx - 1] - 1]:
                        right_bd[heads[head_idx - 1] - 1] = child_idx + 1
                        head_idx = heads[head_idx - 1]
                    else:
                        break

    # (head_word_idx, left_bd_idx, right_bd_idx)
    triplet = []
    # head index should add1, as the root token would be the first token. But not here.
    # [ )  left bdr, right bdr.
    for (parent, left_bdr, right_bdr) in (zip(heads, left_bd, right_bd)):
        if parent != 0:
            if head_in_span:
                triplet.append((left_bdr, right_bdr, parent - 1))
            else:
                triplet.append((left_bdr, right_bdr, heads[parent - 1]))

    return triplet
