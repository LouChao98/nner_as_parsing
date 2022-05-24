import torch
import torch.nn as nn
from src.my_typing import *
from opt_einsum import contract

from . import MLP, Biaffine, Triaffine2, TriaffineLabel
from .modeling_roformer import RoFormerSinusoidalPositionalEmbedding


class BiaffineScorer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        mlp_dropout,
        mlp_activate,
        scale,
        use_position_embedding,
        post_hidden_dim=0,
    ):
        super().__init__()
        self.mlp_dropout = mlp_dropout
        self.mlp1 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp2 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)

        self.post_hidden_dim = post_hidden_dim
        if post_hidden_dim > 0:
            affine_out_dim = post_hidden_dim
            assert post_hidden_dim > 1
            self.activate = nn.LeakyReLU()
            self.linear = nn.Linear(post_hidden_dim, out_dim)
        else:
            affine_out_dim = out_dim
        self.affine = Biaffine(
            hidden_dim, affine_out_dim, bias_x=True, bias_y=out_dim > 1
        )
        self.register_buffer(
            "scale", 1 / torch.tensor(hidden_dim if scale else 1).pow(0.25)
        )
        self.use_position_embedding = use_position_embedding
        if self.use_position_embedding:
            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
                num_positions=512, embedding_dim=hidden_dim
            )

    def forward(self, x: Tensor, x2: Optional[Tensor] = None):
        x2 = x2 if x2 is not None else x
        h1 = self.mlp1(x) * self.scale
        h2 = self.mlp2(x2) * self.scale
        if self.use_position_embedding:
            sinusoidal_pos = self.embed_positions(x.shape[:-1])[None, None, :, :]
            h1, h2 = h1.unsqueeze(1), h2.unsqueeze(1)
            h1, h2 = self.apply_rotary_position_embeddings(sinusoidal_pos, h1, h2)
            h1, h2 = h1.squeeze(1), h2.squeeze(1)

        out = self.affine(h1, h2)
        if self.post_hidden_dim > 0:
            out = out.permute(0, 2, 3, 1)
            out = self.activate(out)
            out = self.linear(out)

            if out.shape[-1] == 1:
                return out.squeeze(-1)
            else:
                return out.permute(0, 3, 1, 2)

        return out

    @staticmethod
    def apply_rotary_position_embeddings(
        sinusoidal_pos, query_layer, key_layer, value_layer=None
    ):
        # https://kexue.fm/archives/8265
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
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
            ).reshape_as(value_layer)
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class FusedTriaffine(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mlp_dropout, mlp_activate):
        super().__init__()
        self.mlp_dropout = mlp_dropout

        self.mlp_span_repr = MLP(in_dim, hidden_dim * 4, mlp_dropout, mlp_activate)
        self.mlp_scoring = MLP(in_dim, hidden_dim * 2, mlp_dropout, mlp_activate)
        self.triaffine_alpha = Triaffine2(hidden_dim, out_dim, True, True)
        self.span_mlp = MLP(hidden_dim, hidden_dim)
        self.weight = nn.Parameter(
            torch.empty(hidden_dim + 1, hidden_dim, hidden_dim + 1)
        )
        nn.init.normal_(self.weight)

    def forward(self, x: Tensor):
        B, L = x.shape[:2]
        hidden = self.mlp_span_repr(x)
        h_in_repr, h_head, h_tail, h_in = torch.chunk(hidden, 4, dim=-1)
        alpha = self.triaffine_alpha(h_tail, h_in, h_head).softmax(-1)

        # B L3 H, B R L1 L2 L3
        # span_repr = (h_in_repr.view(B, 1, 1, 1, L, -1) * alpha.unsqueeze(-1)).sum(-2)  # B x R x L x L x H
        span_repr = torch.einsum("bzh,brxyz->brxyh", h_in_repr, alpha)
        span_repr = self.span_mlp(span_repr)
        x_head, x_tail = torch.chunk(self.mlp_scoring(x), 2, dim=-1)

        x_head = torch.cat((x_head, torch.ones_like(x_head[..., :1])), -1)
        x_tail = torch.cat((x_tail, torch.ones_like(x_head[..., :1])), -1)

        # from pytorch_memlab import MemReporter
        # reporter = MemReporter()
        # reporter.report()
        # input()

        score = contract(
            "bxn,bym,brxyq,nqm->brxy",
            x_head,
            x_tail,
            span_repr,
            self.weight,
            backend="torch",
        )
        return score

    @staticmethod
    def apply_rotary_position_embeddings(
        sinusoidal_pos, query_layer, key_layer, value_layer=None
    ):
        # https://kexue.fm/archives/8265
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
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
            ).reshape_as(value_layer)
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class BiaffineSpan2WordScorer(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_dropout, mlp_activate, scale):
        super().__init__()
        self.mlp_dropout = mlp_dropout
        self.mlp1 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp2 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.affine = Biaffine(hidden_dim, 1, bias_x=True, bias_y=False)

        self.register_buffer(
            "scale", 1 / torch.tensor(hidden_dim if scale else 1).pow(0.25)
        )

    def forward(
        self, x_const: Tensor, x_const2: Optional[Tensor] = None, x_dep: Tensor = None
    ):
        assert x_dep is not None
        x_const2 = x_const2 if x_const2 is not None else x_const

        batch, max_len, hidden = x_const.shape
        x_const = x_const.unsqueeze(1) - x_const2.unsqueeze(2)

        h1 = self.mlp1(x_const.view(batch, -1, hidden)) * self.scale
        h2 = self.mlp2(x_dep) * self.scale

        out = self.affine(h1, h2)
        return out.view(batch, max_len, max_len, -1)


class BiaffineSpan2WordLabeler(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_out, mlp_dropout, mlp_activate, scale):
        super().__init__()
        self.mlp_dropout = mlp_dropout
        self.mlp1 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp2 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.affine = Biaffine(hidden_dim, n_out, bias_x=True, bias_y=False)
        self.n_out = n_out

        self.register_buffer(
            "scale", 1 / torch.tensor(hidden_dim if scale else 1).pow(0.25)
        )

    def forward(
        self, x_const: Tensor, x_const2: Optional[Tensor] = None, x_dep: Tensor = None
    ):
        assert x_dep is not None
        x_const2 = x_const2 if x_const2 is not None else x_const

        batch, max_len, hidden = x_const.shape
        x_const = x_const.unsqueeze(1) - x_const2.unsqueeze(2)

        h1 = self.mlp1(x_const.view(batch, -1, hidden)) * self.scale
        h2 = self.mlp2(x_dep) * self.scale

        out = self.affine(h1, h2)
        return out.view(batch, self.n_out, max_len, max_len, -1)


class TriaffineScorer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        mlp_dropout,
        mlp_activate,
        scale,
        factor=False,
    ):
        super().__init__()
        self.mlp_dropout = mlp_dropout
        self.mlp1 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp2 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.mlp3 = MLP(in_dim, hidden_dim, mlp_dropout, mlp_activate)
        self.factor = factor
        if factor:
            self.affine = TriaffineLabel(
                hidden_dim, hidden_dim, hidden_dim, out_dim, 1.0, 64
            )
        else:
            self.affine = Triaffine2(hidden_dim, out_dim, bias_x=True, bias_y=True)

        self.register_buffer(
            "scale", 1 / torch.tensor(hidden_dim if scale else 1).pow(1 / 6)
        )

    def forward(
        self,
        x: Tensor,
        x2: Optional[Tensor] = None,
        x3: Optional[Tensor] = None,
        diag=False,
    ):
        x2 = x2 if x2 is not None else x
        x3 = x3 if x3 is not None else x

        h1 = self.mlp1(x) * self.scale
        h2 = self.mlp2(x2) * self.scale
        h3 = self.mlp3(x3) * self.scale
        if self.factor:
            return self.affine(h1, h2, h3)
        return self.affine(h2, h3, h1, diag)  # output has shape [o, h1, h2, h3]

