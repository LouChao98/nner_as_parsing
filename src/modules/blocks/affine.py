import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from opt_einsum import contract

class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y`.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f'n_in={self.n_in}'
        if self.n_out > 1:
            s += f', n_out={self.n_out}'
        if self.bias_x:
            s += f', bias_x={self.bias_x}'
        if self.bias_y:
            s += f', bias_y={self.bias_y}'

        return f'{self.__class__.__name__}({s})'

    def reset_parameters(self):
        # nn.init.uniform_(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class DeepBiaffine(nn.Module):
    # NOT the DeepBiaffine in Partially Observed Tree
    def __init__(self, n_in, n_hidden, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_in + bias_x, n_in + bias_y))
        self.linear = nn.Linear(n_hidden, n_out)
        self.activate = nn.LeakyReLU()

        self.reset_parameters()

    def __repr__(self):
        s = f'n_in={self.n_in}'
        if self.n_out > 1:
            s += f', n_out={self.n_out}'
        if self.bias_x:
            s += f', bias_x={self.bias_x}'
        if self.bias_y:
            s += f', bias_y={self.bias_y}'

        return f'{self.__class__.__name__}({s})'

    def reset_parameters(self):
        # nn.init.uniform_(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->bxyo', x, self.weight, y)
        s = self.activate(s)
        s = self.linear(s).squeeze(-1)
        return s


class Triaffine(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.

    https://github.com/wangxinyu0922/Second_Order_Parsing/blob/main/parser/modules/trilinear_attention.py

    """
    def __init__(self, input_size_1, input_size_2, input_size_3, init_std=1., rank=257, factorize=False):
        super(Triaffine, self).__init__()
        self.input_size_1 = input_size_1 + 1
        self.input_size_2 = input_size_2 + 1
        self.input_size_3 = input_size_3 + 1
        self.rank = rank
        self.init_std = init_std
        self.factorize = factorize
        if not factorize:
            self.W = Parameter(torch.Tensor(self.input_size_1, self.input_size_2, self.input_size_3))
        else:
            self.W_1 = Parameter(torch.Tensor(self.input_size_1, self.rank))
            self.W_2 = Parameter(torch.Tensor(self.input_size_2, self.rank))
            self.W_3 = Parameter(torch.Tensor(self.input_size_3, self.rank))
        self.reset_parameters()

    def reset_parameters(self):
        if not self.factorize:
            nn.init.xavier_normal_(self.W)
        else:
            nn.init.xavier_normal_(self.W_1, gain=self.init_std)
            nn.init.xavier_normal_(self.W_2, gain=self.init_std)
            nn.init.xavier_normal_(self.W_3, gain=self.init_std)

    def forward(self, layer1: Tensor, layer2: Tensor, layer3: Tensor):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """

        layer_shape = layer1.size()
        one_shape = list(layer_shape[:2]) + [1]
        ones = layer1.new_ones(one_shape)
        layer1 = torch.cat([layer1, ones], -1)
        layer2 = torch.cat([layer2, ones], -1)
        layer3 = torch.cat([layer3, ones], -1)
        if not self.factorize:
            layer = torch.einsum('nia,abc,njb,nkc->nijk', layer1, self.W, layer2, layer3)
        else:
            layer = torch.einsum('al,nia,bl,njb,cl,nkc->nijk', self.W_1, layer1, self.W_2, layer2, self.W_3, layer3)
        return layer


class TriaffineLabel(nn.Module):
    def __init__(self, input_size_1, input_size_2, input_size_3, num_label, init_std=1., rank=257):
        super(TriaffineLabel, self).__init__()
        self.input_size_1 = input_size_1 + 1
        self.input_size_2 = input_size_2 + 1
        self.input_size_3 = input_size_3 + 1
        self.num_label = num_label
        self.rank = rank
        self.init_std = init_std
        self.W_1 = Parameter(torch.Tensor(self.input_size_1, self.rank))
        self.W_2 = Parameter(torch.Tensor(self.input_size_2, self.rank))
        self.W_3 = Parameter(torch.Tensor(self.input_size_3, self.rank))
        self.W_l = Parameter(torch.Tensor(1, rank, num_label))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W_1, gain=self.init_std)
        nn.init.xavier_normal_(self.W_2, gain=self.init_std)
        nn.init.xavier_normal_(self.W_3, gain=self.init_std)

    def forward(self, layer1: Tensor, layer2: Tensor, layer3: Tensor):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """

        layer_shape = layer1.size()
        one_shape = list(layer_shape[:2]) + [1]
        ones = layer1.new_ones(one_shape)
        layer1 = torch.cat([layer1, ones], -1)
        layer2 = torch.cat([layer2, ones], -1)
        layer3 = torch.cat([layer3, layer2.new_ones(layer3.shape[:-1]).unsqueeze(-1)], -1)
        layer = torch.einsum('al,nia,bl,njb,cl,nkc,blt->ntijk', self.W_1, layer1, self.W_2, layer2, self.W_3, layer3,
                             self.W_l)
        return layer


class Triaffine2(nn.Module):
    r"""
    Triaffine layer for second-order scoring (:cite:`zhang-etal-2020-efficient`, :cite:`wang-etal-2019-second`).

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y`,
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
    """
    def __init__(self, n_in, n_out=1, bias_x=False, bias_y=False, n_feat=None):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.n_feat = n_feat if n_feat is not None else n_in
        if n_feat is not None:
            self.ff = nn.Linear(n_in, n_feat)
        else:
            self.ff = nn.Identity()
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, self.n_feat, n_in + bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f'n_in={self.n_in}'
        if self.n_out > 1:
            s += f', n_out={self.n_out}'
        if self.bias_x:
            s += f', bias_x={self.bias_x}'
        if self.bias_y:
            s += f', bias_y={self.bias_y}'

        return f'{self.__class__.__name__}({s})'

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, z, diag=False):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        if diag is True:
            # w = torch.einsum('bzk,oikj->bozij', z, self.weight)
            # # [batch_size, n_out, seq_len, seq_len, seq_len]
            # s = torch.einsum('bxi,bozij,bxj->bozx', x, w, y)
            # # [batch_size, n_out, seq_len, seq_len, seq_len]
            s = contract('bxi,bzk,oikj,bxj->bozx', x, z, self.weight, y, backend='torch')
            # remove dim 1 if n_out == 1
            s = s.squeeze(1)
        elif diag == 2:
            # w = torch.einsum('bzk,oikj->bozij', z, self.weight)
            # # [batch_size, n_out, seq_len, seq_len, seq_len]
            # s = torch.einsum('bxi,boxij,bxj->box', x, w, y)
            # [batch_size, n_out, seq_len, seq_len, seq_len]
            s = contract('bxi,bxk,oikj,bxj->box', x, z, self.weight, y, backend='torch')
            # remove dim 1 if n_out == 1
            s = s.squeeze(1)
        else:
            # w = torch.einsum('bzk,oikj->bozij', z, self.weight)
            # # [batch_size, n_out, seq_len, seq_len, seq_len]
            # s = torch.einsum('bxi,bozij,byj->bozxy', x, w, y)

            # [batch_size, n_out, seq_len, seq_len, seq_len]
            s = contract('bxi,bzk,oikj,byj->bozxy', x, z, self.weight, y, backend='torch')
            # remove dim 1 if n_out == 1
            s = s.squeeze(1)

        return s

