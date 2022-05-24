import torch
import torch.nn.functional as F
from src.torch_struct import DependencyCRF

from ._eisner_satta import eisner_satta, eisner_satta_v2, eisner_satta_v3


class Dep:
    @staticmethod
    def score(vp, score, gold):
        s_arc = score['arc']
        arcs = gold['arc']
        return s_arc.gather(-1, arcs.unsqueeze(-1)).squeeze(-1)[vp.mask_dep].sum()

    @staticmethod
    def aug(score, gold, margin=1):
        s_arc = score['arc']
        arcs = gold['arc']
        batch_size, max_len, *_ = s_arc.shape
        mask_dep_for_augement = s_arc.new_full((batch_size, max_len, max_len), margin)
        mask_dep_for_augement.scatter_(-1, arcs.unsqueeze(-1), 0)
        return mask_dep_for_augement + s_arc

    @staticmethod
    def head_selection_loss(vp, score, gold):
        s_arc, arcs = score['arc'], gold['arc']
        s_arc_masked, arcs_masked = s_arc[vp.mask_dep], arcs[vp.mask_dep]
        return F.cross_entropy(s_arc_masked, arcs_masked, reduction='sum')
        # return F.cross_entropy(s_arc_masked, arcs_masked)

    @classmethod
    def crf_loss(cls, vp, score, gold, partial=False):
        dist = DependencyCRF(score['arc'].permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
        if partial:
            s_arc = score['arc'].clone()
            gold_mask = vp.partial_mask
            mask_dep_labeled = torch.zeros(s_arc.shape, device=s_arc.device, dtype=torch.bool)
            _mask = torch.full((gold_mask.sum(), s_arc.shape[2]), 1, device=s_arc.device, dtype=torch.bool)
            _mask.scatter_(-1, gold['arc'][gold_mask].unsqueeze(-1), 0)
            mask_dep_labeled[gold_mask] = _mask
            s_arc[mask_dep_labeled] = -1e12
            dist2 = DependencyCRF(s_arc.permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
            return dist.partition.sum() - dist2.partition.sum()
        else:
            return dist.partition.sum() - cls.score(vp, score, gold)

    @classmethod
    def max_margin_loss(cls, vp, score, gold):
        score['arc'] = cls.aug(score, gold, margin=1)
        dist = DependencyCRF(score['arc'].permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
        return dist.max.sum() - cls.score(vp, score, gold)

    @staticmethod
    def label_loss(vp, score, gold, partial=False):
        s_rel = score['rel']
        arcs, rels = gold['arc'], gold['rel']
        mask = vp.mask_dep
        if partial:
            mask = vp.partial_mask
        s_rel_pred, rel_gold = s_rel[mask], rels[mask]
        s_rel_pred = s_rel_pred[torch.arange(len(rel_gold)), arcs[mask]]
        loss = F.cross_entropy(s_rel_pred, rel_gold, reduction='sum')
        # loss = F.cross_entropy(s_rel_pred, rel_gold)
        if partial:
            loss = loss * vp.num_token / mask.sum()
        return loss

    @staticmethod
    def get_pred_arcs_1o(vp, score, mbr=False, dist=None):
        if dist is None:
            s_arc = score['arc']
            if mbr:
                dist = DependencyCRF(s_arc.permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
                s_arc = dist.marginals
                dist = DependencyCRF(s_arc, lengths=vp.real_len, multiroot=False)
            else:
                dist = DependencyCRF(s_arc.permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
        pred_arcs_raw = dist.argmax.nonzero()
        pred_arcs = vp.real_len.new_zeros(vp.batch_size, vp.max_len)
        pred_arcs[pred_arcs_raw[:, 0], pred_arcs_raw[:, 2]] = pred_arcs_raw[:, 1]
        return pred_arcs

        # s_arc = score['arc']
        # if mbr:
        #     dist = DependencyCRF(s_arc.permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
        #     s_arc = dist.marginals.transpose(1, 2)
        # # if local:
        # #     s_arc = s_arc.sigmoid().log()
        # pred_arcs2 = s_arc.argmax(-1)
        # bad = [not istree(seq[1:i + 1], True) for i, seq in zip(vp.real_len.tolist(), pred_arcs2.tolist())]
        # if any(bad):
        #     pred_arcs2[bad] = eisner(s_arc[bad], vp.real_len[bad])

        # return pred_arcs

    @staticmethod
    def get_pred_rels(arc_preds, score):
        return score['rel'].argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def entropy(vp, score):
        dist = DependencyCRF(score['arc'].permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
        return dist.entropy.sum()

    @staticmethod
    def cross_entropy(vp, score, score2):
        dist = DependencyCRF(score['arc'].permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
        dist2 = DependencyCRF(score2['arc'].permute(0, 2, 1).contiguous(), lengths=vp.real_len, multiroot=False)
        return dist.cross_entropy(dist2).sum()

