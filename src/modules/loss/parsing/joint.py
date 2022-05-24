import logging
from math import ceil

import numpy as np
import torch

from src.my_typing import *
from ._eisner_satta import eisner_satta, eisner_satta_for_decode, eisner_satta_entropy
from .const import Const
from .dep import Dep

log = logging.getLogger(__name__)


class FirstOrder:
    @staticmethod
    def partition(vp, score, *args, **kwargs):
        return eisner_satta(score['span'], score['arc'], vp.real_len, *args, **kwargs).sum()

    @classmethod
    def crf_loss(cls, vp, score, gold):
        return cls.partition(vp, score) - Dep.score(vp, score, gold) - Const.score(score, gold)

    @staticmethod
    def max_margin_loss(vp, score, gold):
        s_span = Const.aug(score, gold, margin=1)
        s_arc = Dep.aug(score, gold, margin=1)
        spans, arcs = eisner_satta(s_span, s_arc, vp.real_len, max_marginal=True)
        return (spans * s_span).sum() + (s_arc * arcs).sum() - Dep.score(vp, score, gold) - Const.score(score, gold)

    @staticmethod
    def decode(vp, score, mbr=False, raw_span=False, need_span_head=False):
        if mbr:
            s_span, s_arc = eisner_satta(score['span'], score['arc'], vp.real_len, max_marginal=True)
        else:

            s_span, s_arc = score['span'], score['arc']
        predict_span, predict_arc = eisner_satta_for_decode(s_span,
                                                            s_arc,
                                                            vp.real_len,
                                                            raw=raw_span,
                                                            need_span_head=need_span_head)
        return predict_arc, predict_span

    @staticmethod
    def marginal(vp, score, need_span_head, viterbi=False):
        s_span, s_arc = score['span'].detach().requires_grad_(), score['arc'].detach().requires_grad_()
        if viterbi:
            return eisner_satta(s_span, s_arc, vp.real_len, need_span_head=need_span_head, max_marginal=True)
        with torch.enable_grad():
            logZ, *o = eisner_satta(s_span, s_arc, vp.real_len, need_span_head=need_span_head)
            logZ.sum().backward()
        if need_span_head:
            return s_span.grad, s_arc.grad, o[0].grad
        return s_span.grad, s_arc.grad

    # For partially observed setting, n = seq_len + 1, all are optional.
    #   span_mask:  [b x n x n], [batch x inclusive begin x exclusive end], True for masked.
    #   arc_mask:   [b x n x n], [batch x parent x dependent], True for masked. (diff from score)

    #   span_label: [b x n x n], [batch x inclusive begin x exclusive end], 0 for neg.
    #   arc_label:  [b x n x n], [batch x parent x dependent], 0 for neg. (diff from score)

    @classmethod
    def crf_loss_partially_observed(cls,
                                    vp,
                                    score,
                                    gold,
           
                                    bias_on_diff_head=0.,
                            ):
        if 'arc_mask' in gold:
            s_arc = score['arc'].clone()
            s_arc.masked_fill_(gold['arc_mask'].transpose(-1, -2), -1e12)
        else:
            s_arc = score['arc']
        if 'span_mask' in gold:
            s_span = score['span'].clone()
            s_span.masked_fill_(gold['span_mask'], -1e12)
        else:
            s_span = score['span']

        if bias_on_diff_head != 0:
            # here minus a score instead of maskout = "smooth" in PO TreeCRF
            bias = bias_on_diff_head * gold['span_indicator']
            marginalized = eisner_satta(s_span, s_arc, vp.real_len, bias_on_need_dad=bias).sum()
            logZ = FirstOrder.partition(vp, score)
        else:
            marginalized = eisner_satta(s_span, s_arc, vp.real_len).sum()
            logZ = FirstOrder.partition(vp, score)
        return logZ - marginalized

    @classmethod
    def max_margin_loss_partially_observed(cls, vp, score, gold):
        if 'arc_mask' in gold:
            s_arc = score['arc'].clone()
            s_arc[gold['arc_mask'].transpose(-1, -2)] += 1
        else:
            s_arc = score['arc']
        if 'span_mask' in gold:
            s_span = score['span'].clone()
            s_span[gold['span_mask']] += 1
        else:
            s_span = score['span']
        spans, arcs = eisner_satta(s_span.detach(), s_arc.detach(), vp.real_len, max_marginal=True)
        auged = (spans * s_span).sum() + (s_arc * arcs).sum()

        # we inverse scores to find the tree with lowest prob
        if 'arc_mask' in gold:
            s_arc = -score['arc'].clone()
            s_arc.masked_fill_(gold['arc_mask'].transpose(-1, -2), -1e12)
        else:
            s_arc = -score['arc']
        if 'span_mask' in gold:
            s_span = -score['span'].clone()
            s_span.masked_fill_(gold['span_mask'], -1e12)
        else:
            s_span = -score['span']
        spans, arcs = eisner_satta(s_span.detach(), s_arc.detach(), vp.real_len, max_marginal=True)
        marginalized = -(spans * s_span).sum() - (s_arc * arcs).sum()

        return auged - marginalized

    @classmethod
    def softmax_margin_loss_partially_observed(cls, vp, score, gold, bias_on_diff_head=0.):
        if 'arc_mask' in gold:
            s_arc = score['arc'].clone()
            s_arc[gold['arc_mask'].transpose(-1, -2)] += 1
        else:
            s_arc = score['arc']
        if 'span_mask' in gold:
            s_span = score['span'].clone()
            s_span[gold['span_mask']] += 1
        else:
            s_span = score['span']
        if bias_on_diff_head != 0:
            # bias_on_diff_head > 0
            # * interanlly eisner_satta minus the bias.
            # * we need +bias as cost
            # thus here has a minus
            bias = -bias_on_diff_head * gold['span_indicator']
        else:
            bias = None
        auged_logZ = eisner_satta(s_span, s_arc, vp.real_len, bias_on_need_dad=bias)

        if 'arc_mask' in gold:
            s_arc = score['arc'].clone()
            s_arc.masked_fill_(gold['arc_mask'].transpose(-1, -2), -1e12)
        else:
            s_arc = score['arc']
        if 'span_mask' in gold:
            s_span = score['span'].clone()
            s_span.masked_fill_(gold['span_mask'], -1e12)
        else:
            s_span = score['span']
        spans, arcs = eisner_satta(s_span.detach(), s_arc.detach(), vp.real_len, max_marginal=True)
        marginalized = (spans * s_span).sum() + (s_arc * arcs).sum()
        return auged_logZ - marginalized

    @classmethod
    def hard_em_loss_partially_observed(cls, vp, score, gold):
        if 'arc_mask' in gold:
            s_arc = score['arc'].clone()
            s_arc[gold['arc_mask'].transpose(-1, -2)] = -1e12
        else:
            s_arc = score['arc']
        if 'span_mask' in gold:
            s_span = score['span'].clone()
            s_span[gold['span_mask']] = -1e12
        else:
            s_span = score['span']
        spans, arcs = eisner_satta(s_span.detach(), s_arc.detach(), vp.real_len, max_marginal=True)
        max_score = (spans * s_span).sum() + (s_arc * arcs).sum()
        logZ = FirstOrder.partition(vp, score).sum()
        return logZ - max_score

    @classmethod
    def entropy(cls, vp, score):
        return eisner_satta_entropy(score['span'], score['arc'], vp.real_len).sum()

    @classmethod
    def crf_loss_unfactor_partially_observed(cls, vp, score, gold):
        # if 'arc_mask' in gold:
        #     s_arc = score['arc'].clone()
        #     s_arc.masked_fill_(gold['arc_mask'].transpose(-1, -2), -1e12)
        # else:
        #     s_arc = score['arc']
        if 'span_mask' in gold:
            s_span_head = score['span_head'].clone()
            s_span_head.masked_fill_(gold['span_mask'].unsqueeze(-1), -1e12)
            s_head_span = score['head_span'].clone()
            s_head_span.masked_fill_(gold['span_mask'].unsqueeze(-1), -1e12)
        else:
            s_span_head = score['span_head']
            s_head_span = score['head_span']

        # marginal = eisner_satta_unfactor(score['span_head'].detach().requires_grad_(True), score['head_span'].detach().requires_grad_(True), vp.real_len, max_marginal=True)
        # breakpoint()

        marginalized = eisner_satta_unfactor(s_span_head, s_head_span, vp.real_len).sum()
        logZ = eisner_satta_unfactor(score['span_head'], score['head_span'], vp.real_len).sum()
        return logZ - marginalized

    @staticmethod
    def decode_unfactor(vp, score, mbr=False, raw_span=False, need_span_head=False, marginal_map=False):
        predict_span, predict_arc = eisner_satta_unfactor(score['span_head'],
                                                          score['head_span'],
                                                          vp.real_len,
                                                          decode=True)
        return predict_arc, predict_span
