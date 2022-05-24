from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from omegaconf import MISSING

import src
from src.modules import Const, FirstOrder, Dep
from src.modules.affine_scorer import BiaffineScorer, TriaffineScorer
from src.modules.common import SpanLabeler
from src.modules.loss.sjl_multilabel import multilabel_categorical_crossentropy
from src.my_typing import *
from .base import BratBase
from ..modules.loss.parsing._eisner_satta import eisner_satta
from ..modules.loss.parsing._eisner_satta_exp import (
    eisner_satta_marginal_map,
    eisner_satta_crossentropy2,
)


log = logging.getLogger("task")


@dataclass
class BratNEROneStage2Config(Config):
    # loss_type: support 'none', 'crf', 'max-margin', 'hard_em'
    loss_type: str = "crf"

    # extra term
    partition_reg: float = 0.0
    maximize_entropy: float = 0.0
    exp_opt_entropy: float = 0.0
    entropy_on_all: bool = True
    # for label loss
    label_loss_coeff: float = 1.0
    struct_loss_threshold: float = 0
    neg_weight: float = 0.1
    bias_on_diff_head: float = 0.0
    bias_on_diff_head_impl_version: str = "po"  # po or pr
    marginal_map: bool = False
    unstructured_decode: bool = False
    train_bias: float = 0.0
    decode_bias: float = 0.0
    optimize_arc: bool = 0.0

    # first for struct: num_span, num_span2, batch, token
    # second for label: token, batch,num_span
    loss_reduction_mode: str = "num_span2:token"
    fencepost_mode: bool = "lstm"

    arc_mlp_hidden: int = 500
    arc_post_hidden_dim: int = 0  # 0 to disable
    span_ind_mlp_hidden: int = 100
    span_ind_post_hidden_dim: int = 0  # 0 to disable
    label_mlp_hidden: int = 100
    label_post_hidden_dim: int = 0  # 0 to disable
    label_mode: Optional[List[str]] = None  # None for biaffine
    label_use_head: Optional[str] = None  # None, max, marginal
    label_only_head: Optional[str] = None
    label_pre_marginal: bool = False
    mlp_dropout: float = 0.33
    mlp_activate: bool = True
    scale: bool = False
    positional_biaffine: bool = False
    train_prune_threshold: int = 0  # pad to threshold

    potential_normalize: bool = True
    potential_normalize_var: float = 1.0

    # Following is set automatically according to DataModule.get_vocab_count
    n_span_label: int = MISSING


class BratNEROneStage2(BratBase):
    def __init__(self, encoder: EncoderBase, **cfg):
        super().__init__(encoder, **cfg)
        assert (
            src.g_cfg.runner.loss_reduction_mode == "sum"
        ), "This task implement its own reduction."

        self.cfg = cfg = BratNEROneStage2Config.build(cfg)
        (
            self.struct_loss_reduction,
            self.label_loss_reduction,
        ) = cfg.loss_reduction_mode.split(":")
        assert self.struct_loss_reduction in ("num_span", "num_span2", "batch", "token")
        assert self.label_loss_reduction in ("token", "batch", "num_span")

        if not isinstance(cfg.neg_weight, (float, int)):
            self.set_dynamic_coeff("neg_weight", cfg.neg_weight)
            neg_weight = 1.0
        else:
            neg_weight = cfg.neg_weight
        weight = torch.tensor([neg_weight] + [1] * (cfg.n_span_label - 1))
        self.register_buffer("label_weight", weight)

        self.set_dynamic_coeff("label_loss_coeff", cfg.label_loss_coeff)

        scorer = partial(
            BiaffineScorer,
            mlp_dropout=cfg.mlp_dropout,
            mlp_activate=cfg.mlp_activate,
            scale=cfg.scale,
            use_position_embedding=cfg.positional_biaffine,
        )
        self.arc_scorer: BiaffineScorer = scorer(
            encoder.get_dim("arc"),
            cfg.arc_mlp_hidden,
            1,
            post_hidden_dim=cfg.arc_post_hidden_dim,
        )
        self.span_ind_scorer: BiaffineScorer = scorer(
            encoder.get_dim("span"),
            cfg.span_ind_mlp_hidden,
            2,
            post_hidden_dim=cfg.span_ind_post_hidden_dim,
        )
        if self.cfg.label_mode is not None:
            log.info(f"using label_mode: {self.cfg.label_mode}")
            self.label_scorer = SpanLabeler(
                encoder.get_dim("label"),
                cfg.label_mlp_hidden,
                cfg.n_span_label,
                cfg.label_mode,
                cfg.mlp_dropout,
                cfg.mlp_activate,
            )
        elif self.cfg.label_use_head is None:
            raise NotImplementedError
        else:
            log.info(f"using head: {self.cfg.label_use_head}")
            self.label_scorer: TriaffineScorer = TriaffineScorer(
                encoder.get_dim("label"),
                cfg.label_mlp_hidden,
                cfg.n_span_label,
                cfg.mlp_dropout,
                cfg.mlp_activate,
                cfg.scale,
            )

        if src.g_cfg.datamodule.get("num_label") is not None:
            log.info("using multi_label")
            self.multi_label = True
            self.label_criteria = multilabel_categorical_crossentropy
        else:
            self.multi_label = False
            self.label_criteria = self.ce_loss_adaptor
            log.info("task init done.")

    def ce_loss_adaptor(self, gold, pred):
        return F.cross_entropy(
            pred.permute(0, 3, 1, 2),
            gold.squeeze(-1).expand(-1, -1, pred.shape[2]),
            weight=self.label_weight,
            reduction="none",
        )

    def forward(self, x: TensorDict, vp: VarPool) -> TensorDict:
        dyn_cfg = self.get_dynamic_coeff()
        if dyn_cfg is not None:
            if "neg_weight" in dyn_cfg:
                self.label_weight[0] = dyn_cfg["neg_weight"]
            if "label_loss_coeff" in dyn_cfg:
                self.cfg.label_loss_coeff = dyn_cfg["label_loss_coeff"]

        score = {}

        x_arc = x["arc"][:, :-1]
        score["arc"] = self.arc_scorer(x_arc)

        x_span, x_label = x["span"], x["label"]
        x_span = self.apply_fencepost(x_span, "span")
        score["span_ind"] = self.span_ind_scorer(x_span).permute(
            0, 2, 3, 1
        )  # b x N x N x 2

        x_label = self.apply_fencepost(x_label, "label")
        x_head = x["arc"][:, 1:-1]
        if self.training:
            span_list = vp.span_list  # in y
            if self.cfg.train_prune_threshold > 0:
                span_list = span_list.tolist()
                with torch.no_grad():
                    L = score["span_ind"].shape[1]
                    span_ind = score["span_ind"].clone()
                    span_ind = span_ind.masked_fill_(
                        torch.tril(span_ind.new_ones(L, L, dtype=torch.bool)).unsqueeze(
                            -1
                        ),
                        -1e9,
                    )
                    _diff = span_ind[..., 0] - span_ind[..., 1]
                    _diff.masked_fill_(_diff <= 0, 1000)
                    span_ind[..., 1] += (
                        _diff.view(-1, L * L).min(1)[0].view(-1, 1, 1) + 1e-3
                    )
                    span_ind = span_ind.argmax(-1)
                    predict_span = [
                        span_ind[i].nonzero() for i in range(len(span_list))
                    ]
                    predict_span = [p.tolist() for p in predict_span]
                new_span = []
                max_num = 0
                for bid in range(len(span_list)):
                    current_spans = span_list[bid]
                    current_pred_spans = predict_span[bid]
                    idx = 0
                    for idx in range(len(current_spans)):
                        if current_spans[idx][1] == 0:
                            break
                    if idx < self.cfg.train_prune_threshold:
                        num = self.cfg.train_prune_threshold - idx
                        if len(current_pred_spans) < num:
                            new_current_spans = current_spans[:idx] + current_pred_spans
                        else:
                            new_current_spans = current_spans[:idx] + random.sample(
                                current_pred_spans, num
                            )
                    else:
                        new_current_spans = current_spans
                    if len(new_current_spans) > max_num:
                        max_num = len(new_current_spans)
                    new_span.append(new_current_spans)
                for current_spans in new_span:
                    if len(current_spans) < max_num:
                        current_spans.extend([[0, 0]] * (max_num - len(current_spans)))
                if max_num == 0:
                    span_list = torch.empty(len(new_span), 0, 2, device=x_span.device)
                    breakpoint()
                else:
                    span_list = torch.tensor(new_span, device=x_span.device)
                # print(span_list.shape)
                vp.span_list = span_list

            left = x_label.gather(
                1, span_list[..., 0, None].expand(-1, -1, x_label.shape[-1])
            )
            right = x_label.gather(
                1, span_list[..., 1, None].expand(-1, -1, x_label.shape[-1])
            )
            if self.cfg.label_pre_marginal:
                score["label_input"] = (x_head, left, right)
            else:
                score["label"] = self.label_scorer(
                    x_head, left, right, diag=True
                ).permute(0, 3, 2, 1)
        else:
            score["label_input"] = (x_head, x_label)
        score = self.normalize_potential(score, fields=["arc", "span_ind"])
        return score

    def loss(
        self, x: TensorDict, gold: InputDict, vp: VarPool
    ) -> Tuple[Tensor, TensorDict]:
        if self.struct_loss_reduction == "num_span":
            struct_normalize = sum(gold["num_span"])
        elif self.struct_loss_reduction == "num_span2":
            struct_normalize = sum(gold["num_span2"])
        elif self.struct_loss_reduction == "batch":
            struct_normalize = vp.batch_size
        else:  # self.struct_loss_reduction == 'token':
            struct_normalize = vp.num_token

        if self.label_loss_reduction == "token":
            label_normalize = vp.num_token
        elif self.struct_loss_reduction == "batch":
            label_normalize = vp.batch_size
        else:  # self.struct_loss_reduction == 'num_span':
            label_normalize = sum(gold["num_span"])

        loss = {}
        if struct_normalize > 0:
            x_span_ob = x["span_ind"].clone()
            x_span_ob[..., 0] -= self.cfg.train_bias
            x_span_ob.masked_fill_(gold["span_mask"].unsqueeze(-1), -1e9)
            x_span_ob[..., 0].masked_fill_(gold["span_indicator"], -1e9)
            x_span_ob[..., 1].masked_fill_(
                ~gold["span_mask"] & ~gold["span_indicator"], -1e9
            )
            x_span_ob = x_span_ob.logsumexp(-1)
            x_span_all = x["span_ind"].logsumexp(-1)

            if self.cfg.optimize_arc > 0:
                _, dep_margin = FirstOrder.marginal(
                    vp, {"span": x_span_ob, "arc": x["arc"]}, need_span_head=False
                )
                loss["arc"] = (
                    self.cfg.optimize_arc
                    * Dep.cross_entropy(vp, {"arc": dep_margin}, {"arc": x["arc"]})
                    / struct_normalize
                )

            if self.cfg.bias_on_diff_head != 0:
                bias = self.cfg.bias_on_diff_head * gold["span_indicator"]
            else:
                bias = None

            if self.cfg.bias_on_diff_head_impl_version == "pr":
                loss["bias"] = (
                    eisner_satta_crossentropy2(
                        x_span_all, x["arc"], bias, vp.real_len
                    ).sum()
                    / struct_normalize
                )
                # loss['bias'] = EisnerSattaKL.apply(x_span_all, x['arc'], bias, vp.real_len)
                bias = None

            marginalized = eisner_satta(
                x_span_ob, x["arc"], vp.real_len, bias_on_need_dad=bias
            ).sum()
            logZ = eisner_satta(x_span_all, x["arc"], vp.real_len).sum()
            loss["struct"] = logZ - marginalized
            loss["struct"] /= struct_normalize

            if self.cfg.entropy_on_all:
                x["span"] = x_span_all
            else:
                x["span"] = x_span_ob  # x_span_all  # be carefull
        else:
            x["span"] = x["span_ind"].logsumexp(-1)

        if self.training:
            _, dep_margin, span_head_margin = FirstOrder.marginal(
                vp, x, need_span_head=True, viterbi=self.cfg.label_use_head == "max"
            )
            gold_span = vp.span_list

            batch_arange = (
                torch.arange(len(gold_span))
                .unsqueeze(-1)
                .expand(-1, gold_span.shape[1])
                .flatten()
            )
            gold_label = gold["span_label"][
                batch_arange, gold_span[..., 0].flatten(), gold_span[..., 1].flatten()
            ]
            span_head_margin = span_head_margin[
                batch_arange, gold_span[..., 0].flatten(), gold_span[..., 1].flatten()
            ]
            span_mask = (gold_span.sum(-1) == 0).flatten()
            # if not (span_head_margin[~span_mask].sum(-1) > 0.999).all():
            #     breakpoint()  # this proof masking must lead to the occurance of entitis.

            if self.cfg.label_pre_marginal:
                x_head, left, right = x["label_input"]
                # batch len hidden * num(batch*per) len
                head = (
                    x_head.unsqueeze(1)
                    * span_head_margin.view(
                        len(x_head), -1, span_head_margin.shape[-1], 1
                    )
                ).sum(2)
                s_label = self.label_scorer(head, left, right, diag=2).permute(0, 2, 1)
                label_loss = self.label_criteria(
                    gold_label.view(*s_label.shape), s_label
                )
                label_loss = (label_loss * ~span_mask.view(len(x_head), -1)).sum()
            else:
                span_head_margin.masked_fill_(span_mask.unsqueeze(-1), 0)
                label_loss = self.label_criteria(
                    gold_label.view(*x["label"].shape[:2], 1, -1), x["label"]
                )
                span_head_margin = span_head_margin.view(*x["label"].shape[:2], -1)
                label_loss = (label_loss * span_head_margin).sum()
            loss["label"] = self.cfg.label_loss_coeff * label_loss
            loss["label"] /= label_normalize + 1e-9

        if self.cfg.maximize_entropy > 0 and vp.max_len < 100:
            loss["neg_entropy"] = (
                -FirstOrder.entropy(vp, x) / vp.num_token * self.cfg.maximize_entropy
            )

        if self.cfg.exp_opt_entropy > 0 and vp.max_len < 100:
            loss["exp_entropy"] = (
                (Dep.entropy(vp, x) - Const.entropy(vp, x))
                / vp.num_token
                * self.cfg.exp_opt_entropy
            )

        if self.cfg.partition_reg > 0:
            loss["partition"] = FirstOrder.partition(vp, x).sum()

        if (t := self.cfg.struct_loss_threshold) > 0:
            total_loss = sum(
                max(value, t) if key == "struct" else value
                for key, value in loss.items()
            )
        else:
            total_loss = sum(loss.values())

        return total_loss, loss

    def decode(self, x: TensorDict, vp: VarPool) -> TensorDict:
        with torch.no_grad():
            x["span_ind"][..., 0] += self.cfg.decode_bias
        x["span"], pred_label = x["span_ind"].max(-1)
        pred_label = pred_label.cpu().numpy()
        if self.cfg.marginal_map:
            x_head, x_label = x["label_input"]
            c, _, span_head_margin = eisner_satta_marginal_map(
                x["span"], x["arc"], vp.real_len
            )
            # c2, _, gold = eisner_satta(x['span'], x['arc'], vp.real_len, max_marginal=True, need_span_head=True)
            # if not (span_head_margin.max(3)[1] == gold.max(3)[1]).all():
            #     breakpoint()
            raw_spans = c.nonzero().tolist()
            pred_span = [[] for _ in range(len(x["span"]))]
            for b, start, end in raw_spans:
                if len(pred_span[b]) == 100:
                    log.debug("Too many predicted spans.")
                    continue
                if pred_label[b, start, end] == 1:
                    pred_span[b].append((start, end))
            # debug marginal MAP
            # span_mask = np.zeros((batch_size, seq_len, seq_len), dtype=np.bool8)
            # for bidx, left, right in raw_spans:
            #     span_mask[bidx, :left, left + 1:right] = True
            #     span_mask[bidx, left + 1:right, right + 1:] = True
            # span_mask = torch.from_numpy(span_mask).to(x['span'].device)
            # _score = {**x}
            # _score['span'] = _score['span'].masked_fill(span_mask, -1e9)
            # *_, _marginal = FirstOrder.marginal(vp, _score, need_span_head=True)

            pred_span_max = max(len(inst) for inst in pred_span)
            pred_span_tensor = torch.zeros(
                len(x["span"]), pred_span_max, 2, dtype=torch.long
            )
            for bidx, inst in enumerate(pred_span):
                if len(inst) > 0:
                    pred_span_tensor[bidx, : len(inst)] = torch.tensor(inst)
            pred_span_tensor = pred_span_tensor.to(x["span"].device)

            left = x_label.gather(
                1, pred_span_tensor[..., 0, None].expand(-1, -1, x_label.shape[-1])
            )
            right = x_label.gather(
                1, pred_span_tensor[..., 1, None].expand(-1, -1, x_label.shape[-1])
            )
            batch_arange = (
                torch.arange(len(pred_span_tensor))
                .unsqueeze(-1)
                .expand(-1, pred_span_tensor.shape[1])
                .flatten()
            )
            span_head_margin = span_head_margin[
                batch_arange,
                pred_span_tensor[..., 0].flatten(),
                pred_span_tensor[..., 1].flatten(),
            ]
            span_head_margin = span_head_margin.view(
                len(x_head), -1, span_head_margin.shape[-1], 1
            )
            if self.cfg.label_pre_marginal:
                head = (x_head.unsqueeze(1) * span_head_margin).sum(
                    2
                )
                s_label = self.label_scorer(head, left, right, diag=2).permute(0, 2, 1)
            else:
                s_label = self.label_scorer(x_head, left, right, diag=True).permute(
                    0, 3, 2, 1
                )
                s_label = (s_label * span_head_margin).sum(2)
        else:
            # pred_arcs, pred_span = FirstOrder.decode(
            #     vp,
            #     x,
            #     False,
            #     need_span_head=self.cfg.label_use_head is not None,
            # )  #marginal_map=self.cfg.marginal_map)

            pred_span, pred_arcs = eisner_satta(
                x["span"],
                x["arc"],
                vp.real_len,
                decode=True,
                need_span_head=True,
                bias_on_need_dad=self.cfg.bias_on_diff_head
                * torch.tensor(pred_label).to(x["span"].device),
            )
            filtered_span = []
            for pred, mask in zip(pred_span, pred_label):
                filtered_pred = [s for s in pred if mask[s[0], s[1]] == 1]
                filtered_span.append(filtered_pred)
            pred_span = filtered_span

            pred_span_max = max(len(inst) for inst in pred_span)
            pred_span_tensor = torch.zeros(
                len(x["span"]), pred_span_max, 3, dtype=torch.long
            )
            for bidx, inst in enumerate(pred_span):
                if len(inst) > 0:
                    pred_span_tensor[bidx, : len(inst)] = torch.tensor(inst)
            pred_span_tensor = pred_span_tensor.to(x["span"].device)
            x_head, x_label = x["label_input"]
            head = x_head.gather(
                1, pred_span_tensor[..., 2, None].expand(-1, -1, x_head.shape[-1])
            )
            left = x_label.gather(
                1, pred_span_tensor[..., 0, None].expand(-1, -1, x_label.shape[-1])
            )
            right = x_label.gather(
                1, pred_span_tensor[..., 1, None].expand(-1, -1, x_label.shape[-1])
            )
            s_label = (
                self.label_scorer(head, left, right, diag=2).permute(0, 2, 1).cpu()
            )

        pred_charts = []
        pred_headed_charts = []

        for i, inst in enumerate(pred_span):
            labels_inst = []
            labels_headed_inst = []
            for sidx, span in enumerate(inst):
                span_score = s_label[i, sidx]
                labels = torch.where(span_score > 0.)[0].tolist()
                if 0 in labels:
                    continue
                labels_inst.extend((span[0], span[1], l) for l in labels)
                if len(span) == 2:
                    labels_headed_inst.extend((span[0], span[1], 0, l) for l in labels)
                else:
                    labels_headed_inst.extend(
                        (span[0], span[1], span[2], l) for l in labels
                    )
            pred_charts.append(labels_inst)
            pred_headed_charts.append(labels_headed_inst)

        out = {
            "span": pred_charts,
            "headed_span": pred_headed_charts,
        } 
        if "raw_span" in vp and self.cfg.label_use_head is None:
            gold_span = [
                [(l, r) for _, l, r in spans.values()] for spans in vp.raw_span
            ]
            out["span_gold_boundary"] = Const.get_pred_charts(gold_span, x)
        return out

    def write_prediction(
        self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]
    ):
        label_vocab = vocabs["span_label"]
        for i, length in enumerate(dataset["seq_len"].content):
            word, chart = dataset[i]["raw_words"], predicts["headed_span"][i]
            assert len(word) == length - 2
            s.write(" ".join(word))
            s.write("\n")
            # if self.cfg.label_use_head:
            s.write(
                "|".join(
                    [
                        f"{l},{r}({h}) {label_vocab.to_word(label)}"
                        for l, r, h, label in chart
                        if label != 0
                    ]
                )
            )
            # else:
            # s.write('|'.join([f'{l},{r} {label_vocab.to_word(label)}' for l, r, label in chart if label != 0]))
            s.write("\n")
            # s.write(' '.join(map(str, predicts['arc'][i][1:length - 1])))
            s.write("\n\n")
        return s

    # def set_varpool(self, vp: VarPool) -> VarPool:
    #     vp = super().set_varpool(vp)

    #     def _to_mask_label(seq_len, max_len):
    #         # return: [batch, inclusive left, exclusive right]
    #         m: Tensor = seq_len_to_mask(seq_len - 1, max_len - 1)
    #         m = m.unsqueeze(-1) * m.unsqueeze(-2)
    #         m *= torch.triu(torch.ones(max_len - 1, max_len - 1, device=m.device, dtype=torch.bool), 1)
    #         return m

    #     def _to_valid_prediction(seq_len, max_len):
    #         return seq_len_to_mask(seq_len - 1, max_len - 1)

    #     vp.add_lazy('mask', 'mask_dep', lambda x: x[:, :-1])
    #     vp.add_lazy(['seq_len', 'max_len'], 'mask_label', _to_mask_label)
    #     vp.add_lazy(['seq_len', 'max_len'], 'valid_predict', _to_valid_prediction)  # mask for prediction
    #     return vp
