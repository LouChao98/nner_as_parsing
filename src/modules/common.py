import torch
import torch.nn as nn

from src.my_typing import *
from . import MLP, Biaffine
import numpy as np


class SquareNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mode, mlp_dropout, mlp_activate):
        super().__init__()
        self.mlp_dropout = mlp_dropout
        self.n_out = out_dim
        self.mode = [m.split(":")[0] for m in mode]
        self.extra_mode = [m.split(":")[1:] for m in mode]
        self.mlp1 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp2 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)

        extra_mode_set = set(i for em in self.extra_mode for i in em)
        # init position encoding
        if "rotary" in extra_mode_set:
            self.rotary_pemb = RotaryPositionalEmbedding(512, hidden_dim)
        if "sin" in extra_mode_set:
            self.sin_pemb = SinusoidalPositionalEmbedding(512, hidden_dim)
        if "learnable" in extra_mode_set:
            self.learnable_pemb = LearnedPositionalEmbedding(512, hidden_dim)

        # init nn
        if "biaffine" in self.mode:
            self.affine = Biaffine(hidden_dim, out_dim, bias_x=True, bias_y=out_dim > 1)
        n_basic_mode = len(
            set(self.mode).intersection({"i-j", "j-i", "i+j", "i", "j", "mean"})
        )
        if n_basic_mode > 0:
            self.ff = nn.Linear(hidden_dim * n_basic_mode, out_dim)

    def forward(self, x: Tensor, x2: Optional[Tensor] = None):
        x2 = x2 if x2 is not None else x

        x = self.mlp1(x)
        x2 = self.mlp2(x2)
        batch, max_len, hidden = x.shape

        feat, out = [], []
        for m, em in zip(self.mode, self.extra_mode):
            for em_item in em:
                if em_item == "rotary":
                    x, x2 = self.rotary_pemb(x, x2)
                elif em_item == "sin":
                    x, x2 = self.sin_pemb(x, x2)
                elif em_item == "learnable":
                    x, x2 = self.learnable_pemb(x, x2)
            if m == "biaffine":
                _x = self.affine(x, x2).permute(0, 3, 1, 2)
                out.append(_x)
            elif m == "i-j":
                feat.append(x.unsqueeze(1) - x2.unsqueeze(2))
            elif m == "j-i":
                feat.append(-x.unsqueeze(1) + x2.unsqueeze(2))
            elif m == "i+j":
                feat.append((x2.unsqueeze(1) + x.unsqueeze(2)) / 2)
            elif m == "i":
                feat.append(x.unsqueeze(1).expand(batch, max_len, max_len, hidden))
            elif m == "j":
                feat.append(x2.unsqueeze(2).expand(batch, max_len, max_len, hidden))
            elif m == "mean":
                cum = torch.cumsum(x, dim=1)
                span_sum = cum.unsqueeze(1).repeat(1, max_len, 1, 1)
                tmp = cum.unsqueeze(2)
                span_sum[:, 1:] -= tmp[:, :-1]

                size = cum.new_ones(1, max_len, 1)
                size_cum = torch.cumsum(size, dim=1)
                size_span_sum = size_cum.unsqueeze(1).repeat(1, max_len, 1, 1)
                size_tmp = size_cum.unsqueeze(2)
                size_span_sum[:, 1:] -= size_tmp[:, :-1]

                span_sum /= size_span_sum + 1e-9
                feat.append(span_sum)

        feat = torch.cat(feat, dim=-1)
        if out:
            out = torch.cat([out, self.ff(feat)], dim=-1)
        else:
            out = self.ff(feat)
        out = out.view(batch, max_len, max_len, -1).permute(0, 3, 1, 2)
        if self.n_out == 1:
            assert out.shape[-1] == 1
            out = out.squeeze(-1)
        return out


class RotaryPositionalEmbedding(nn.Module):
    # https://kexue.fm/archives/8265

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def _forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer):
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        return query_layer

    def forward(self, x: Tensor, x2: Tensor):
        sinusoidal_pos = self._forward(x.shape[:-1])[None, None, :, :]
        if x is not x2:
            x = self.apply_rotary_position_embeddings(
                sinusoidal_pos, x.unsqueeze(1)
            ).squeeze(1)
            x2 = self.apply_rotary_position_embeddings(
                sinusoidal_pos, x2.unsqueeze(1)
            ).squeeze(1)
        else:
            x = x2 = self.apply_rotary_position_embeddings(
                sinusoidal_pos, x.unsqueeze(1)
            ).squeeze(1)
        return x, x2


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # WHY?
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def _forward(self, input):
        positions = torch.arange(input.shape[1], device=input.device)
        return super().forward(positions).unsqueeze(0)

    def forward(self, x: Tensor, x2: Tensor):
        pos = self._forward(x)
        if x is not x2:
            x = x + pos
            x2 = x2 + pos
        else:
            x = x2 = x + pos
        return x, x2


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(
                f"odd embedding_dim {embedding_dim} not supported"
            )
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0 : dim // 2] = torch.FloatTensor(
            np.sin(position_enc[:, 0::2])
        )  # This line breaks for odd n_pos
        out[:, dim // 2 :] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def _forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions).unsqueeze(0)

    def forward(self, x: Tensor, x2: Tensor):
        pos = self._forward(x)
        if x is not x2:
            x = x + pos
            x2 = x2 + pos
        else:
            x = x2 = x + pos
        return x, x2


class SpanLabeler(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mode, mlp_dropout, mlp_activate):
        super().__init__()
        self.mlp_dropout = mlp_dropout
        self.n_out = out_dim
        self.mode = mode
        self.mlp1 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp2 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.ff = nn.Linear(hidden_dim * len(mode), out_dim)

    def forward(self, x_const: Tensor, x_const2: Optional[Tensor] = None):
        x_const2 = x_const2 if x_const2 is not None else x_const

        x_const = self.mlp1(x_const)
        x_const2 = self.mlp2(x_const2)
        batch, max_len, hidden = x_const.shape

        feat = []
        for m in self.mode:
            if m == "i-j":
                feat.append(x_const.unsqueeze(1) - x_const2.unsqueeze(2))
            elif m == "j-i":
                feat.append(-x_const.unsqueeze(1) + x_const2.unsqueeze(2))
            elif m == "i+j":
                feat.append((x_const2.unsqueeze(1) + x_const.unsqueeze(2)) / 2)
            elif m == "i":
                feat.append(
                    x_const.unsqueeze(1).expand(batch, max_len, max_len, hidden)
                )
            elif m == "j":
                feat.append(
                    x_const2.unsqueeze(2).expand(batch, max_len, max_len, hidden)
                )
            elif m == "mean i":
                cum = torch.cumsum(x_const, dim=1)
                span_sum = cum.unsqueeze(1).repeat(1, max_len, 1, 1)
                tmp = cum.unsqueeze(2)
                span_sum[:, 1:] -= tmp[:, :-1]

                size = cum.new_ones(1, max_len, 1)
                size_cum = torch.cumsum(size, dim=1)
                size_span_sum = size_cum.unsqueeze(1).repeat(1, max_len, 1, 1)
                size_tmp = size_cum.unsqueeze(2)
                size_span_sum[:, 1:] -= size_tmp[:, :-1]

                span_sum /= size_span_sum + 1e-9
                feat.append(span_sum)

        feat = torch.cat(feat, dim=-1)
        out = self.ff(feat)
        return out.view(batch, max_len, max_len, -1).permute(0, 3, 1, 2)

