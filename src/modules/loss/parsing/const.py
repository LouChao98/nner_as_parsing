import torch.nn.functional as F
from torch import Tensor

from ._cyk import cyk
from src.torch_struct import TreeCRF

class Const:
    @staticmethod
    def partition(vp, score):
        return cyk(s_span=score['span'], lens=vp.real_len).sum()

    @staticmethod
    def score(score, gold):
        s_span = score['span']
        true_span_mask = gold['true_span_mask']
        return (s_span[true_span_mask]).sum()

    @staticmethod
    def aug(score, gold, margin):
        s_span = score['span'].clone()
        true_span_mask = gold['true_span_mask']
        s_span += margin
        s_span[true_span_mask] -= margin
        return s_span

    @staticmethod
    def crf_loss(vp, score, gold):
        return (Const.partition() - Const.score(score, gold))

    @staticmethod
    def bce_loss(vp, score, gold):
        s_span = score['span']
        mask_const = vp.mask_const
        span_mask = gold['true_span_mask']
        return F.binary_cross_entropy_with_logits(s_span[mask_const], span_mask[mask_const].float())

    @staticmethod
    def label_loss(score, gold):
        s_label = score['label']
        true_span_mask = gold['true_span_mask']
        charts = gold['chart']
        return F.cross_entropy(s_label[true_span_mask], charts[true_span_mask])

    @staticmethod
    def get_pred_spans(vp, score, raw=False):
        s_span = score['span']
        span_preds = cyk(s_span=s_span, lens=vp.real_len, decode=True, raw_decode=raw)
        return span_preds

    @staticmethod
    def get_pred_charts(span_preds, score, has_head=False):

        label_preds = score['label'].argmax(-1).tolist()
        if has_head:
            chart_preds = [[(i, j, h, labels[i][j]) for i, j, h in spans] for spans, labels in
                           zip(span_preds, label_preds)]
        else:
            chart_preds = [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in
                           zip(span_preds, label_preds)]
        return chart_preds

    @staticmethod
    def get_pred_10_charts(span_preds, score):
        label_preds = (score['label'] > 0).cpu().numpy()
        chart_preds = [[(i, j) for i, j in spans if labels[i, j]] for spans, labels in zip(span_preds, label_preds)]
        return chart_preds

    @classmethod
    def crf_loss_partially_observed(cls, vp, score, gold):
        if 'span_mask' in gold:
            s_span: Tensor = score['span'].clone()
            s_span.masked_fill_(gold['span_mask'], -1e12)
        else:
            s_span = score['span']
        marginalized = cyk(s_span, vp.real_len).sum()
        logZ = Const.partition(vp, score).sum()
        return logZ - marginalized

    @staticmethod
    def entropy(vp, score):
        dist = TreeCRF(score['span'][:, :-1, 1:].unsqueeze(-1), lengths=vp.real_len)
        return dist.entropy.sum()
