def stripe_version5(x, n, w=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[2:])), stride=new_stride, storage_offset=0)


def stripe_step(x, n, w, dim=0):
    """在dim上，每个n重复输出w"""
    assert x.is_contiguous()
    stride = list(x.stride())
    stride.insert(dim, stride[dim])
    size = list(x.size())
    size[dim:dim + 1] = [n, w]
    return x.as_strided(size=size, stride=stride)


def stripe_version2(x, n, w, offset=(0, 0), dim=1):
    """选n,w. 并且返回在中间的w+1."""
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(x.shape[0], n, w, w + 1, *list(x.shape[4:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def diagonal_copy_v2(x, y, w, lower=False):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                 stride=new_stride,
                 storage_offset=w * stride[2] if not lower else w * stride[1]).copy_(y)


def diagonal_v2(x, w, lower=False):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2] if not lower else w*stride[1]
                     )

# ======================================================
# for const_sibling
def stripe_version3(x, n, w, offset=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, w, *list(x.shape[3:])), stride=new_stride, storage_offset=0)


def stripe_need_child(x, n, w1, w2, start, end, headstart, childstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3] + stride[4])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    return x.as_strided(size=(x.shape[0], n, w1, w2, *list(x.shape[5:])),
                        stride=new_stride,
                        storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3] +
                        childstart * stride[4])


# used in lexicalized-pcfg.
def stripe_version4(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3] + stride[4]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(x.shape[0], n, w, w + 1, w + 1, *list(x.shape[5:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def diagonal_copy_v2_for_split_point(x, y, w, lower=False):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3] + stride[4])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], seq_len - w, w, w, *list(x.shape[5:])),
                 stride=new_stride,
                 storage_offset=w * stride[2] if not lower else w * stride[1]).copy_(y)


# ======================================================
# for cyk
def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_compose(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    # new_stride.append(stride)
    stride[1] = (seq_len + 1) * numel + stride[3]
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    return x.as_strided(size=(x.shape[0], n, w - 1), stride=new_stride, storage_offset=(1) * numel + w * stride[3])


def diagonal_copy_(x, y, w):
    # size of x: (batch, N, N, nt)
    # size of y: (batch, N, nt)
    # the function aims to copy y to the diagonal of x (dim1 and dim2) without any copy of tensor.
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w), stride=new_stride, storage_offset=w * stride[2]).copy_(y)


def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=w * stride[2])
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w), stride=new_stride, storage_offset=w * stride[2])


# ======================================================
# for const_grand


def stripe_02(x, n, w, offset=(0, 0, 0)):
    # x: [seq_len, seq_len, seq_len, ...]
    assert x.is_contiguous(), 'x must be contiguous, or write on new view will lost.'
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel1, numel2 = list(x.stride()), x[0, 0].numel(), x[0, 0, 0].numel()
    stride[0] = (seq_len + 1) * numel1 + numel2
    del stride[1]
    return x.as_strided(size=(n, w, *x.shape[3:]),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel1 + offset[2] * numel2)

def stripe_3d(x, n, w, offset=(0, 0), dim=1):
    """选n,w. 并且返回在中间的w+1. NNNNB (unlike stripe_version2)"""
    x, seq_len = x.contiguous(), x.size(0)
    stride = list(x.stride())
    numel = stride[1]
    stride[0] = (seq_len + 1) * numel + stride[2]
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, w + 1, *list(x.shape[3:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)
