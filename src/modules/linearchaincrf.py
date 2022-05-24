# Based on flair's implementation, only for debug
# torch-struct is recommanded for training/inference.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


class FlairLinearChainCRF(nn.Module):
    def __init__(self, ntag):
        super().__init__()
        self.ntag = ntag + 2
        self.start_id = ntag
        self.stop_id = ntag + 1
        self.transitions = nn.Parameter(torch.zeros(self.ntag, self.ntag))
        self.transitions.data[self.start_id, :] = -1e12
        self.transitions.data[:, self.stop_id] = -1e12

    def _score_sentence(self, feats, tags, lens_):
        start = torch.tensor([self.start_id], device=feats.device)
        start = start[None, :].repeat(tags.shape[0], 1)
        stop = torch.tensor([self.stop_id], device=feats.device)
        stop = stop[None, :].repeat(tags.shape[0], 1)
        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)
        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = self.stop_id
        score = torch.empty(feats.shape[0], device=feats.device)
        for i in range(feats.shape[0]):
            r = torch.arange(lens_[i], device=feats.device)
            score[i] = torch.sum(self.transitions[pad_stop_tags[i, :lens_[i] + 1], pad_start_tags[i, :lens_[i] + 1]]) \
                + torch.sum(feats[i, r, tags[i, :lens_[i]]])
        return score

    def _calculate_loss(self, features: torch.tensor, tags, lengths) -> float:
        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, tags, lengths)
        score = gold_score - forward_score
        return score.mean()

    def _forward_alg(self, feats, lens_, T=1):

        init_alphas = torch.FloatTensor(self.ntag).fill_(-1e12)
        init_alphas[self.start_id] = 0.0
        forward_var = feats.new_zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
        )
        forward_var[:, 0, :] = init_alphas.repeat(feats.shape[0], 1)
        transitions = self.transitions.repeat(feats.shape[0], 1, 1)
        # breakpoint()
        if T != 1:
            transitions = transitions / T
            feats = feats / T

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]
            tag_var = (emit_score[:, :, None].repeat(1, 1, transitions.shape[2]) + transitions +
                       forward_var[:, i, :][:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1))
            max_tag_var, _ = torch.max(tag_var, dim=2)
            tag_var = tag_var - max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2])
            agg_ = torch.logsumexp(tag_var, dim=2)
            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_
            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        terminal_var = forward_var + self.transitions[self.stop_id].repeat(forward_var.shape[0], 1) / T
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _backward_alg(self, feats, lens_, T=1):
        bw_transitions = self.transitions.transpose(0, 1)
        reversed_feats = torch.zeros_like(feats)

        for i, feat in enumerate(feats):
            # m * d -> k * d, reverse over tokens -> m * d
            reversed_feats[i][:lens_[i]] = feat[:lens_[i]].flip([0])
            # reverse_feats[i][:lens_[i]] = feat[:lens_[i]].filp(0)

        init_alphas = torch.full(self.ntag, fill_value=-1e12)
        init_alphas[self.stop_id] = 0.0

        forward_var = feats.new_zeros(
            reversed_feats.shape[0],
            reversed_feats.shape[1] + 1,
            reversed_feats.shape[2],
        )
        forward_var[:, 0, :] = init_alphas[None, :].repeat(reversed_feats.shape[0], 1)
        transitions = bw_transitions.repeat(reversed_feats.shape[0], 1, 1)

        if T != 1:
            transitions = transitions / T
            reversed_feats = reversed_feats / T

        for i in range(reversed_feats.shape[1]):
            if i == 0:
                emit_score = torch.zeros_like(reversed_feats[:, 0, :])
            else:
                emit_score = reversed_feats[:, i - 1, :]
            tag_var = (emit_score[:, None, :].repeat(1, transitions.shape[2], 1) + transitions +
                       forward_var[:, i, :][:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1))

            max_tag_var, _ = torch.max(tag_var, dim=2)
            tag_var = tag_var - max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2])
            agg_ = torch.logsumexp(tag_var, dim=2)
            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_
            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        terminal_var = forward_var + bw_transitions[self.start_id].repeat(forward_var.shape[0], 1) \
            + reversed_feats[range(reversed_feats.shape[0]), lens_ - 1, :] / T
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _viterbi_decode(self, feats, all_scores: bool = False):
        backpointers = []
        backscores = []

        init_vvars = torch.full((1, self.ntag), fill_value=-1e12, device=feats.device)
        init_vvars[0][self.start_id] = 0
        forward_var = init_vvars
        for i, feat in enumerate(feats):
            next_tag_var = (forward_var.view(1, -1).expand(self.ntag, self.ntag) + self.transitions)
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)
        terminal_var = (forward_var + self.transitions[self.stop_id])
        terminal_var.detach()[self.stop_id] = -1e12
        terminal_var.detach()[self.start_id] = -1e12
        best_tag_id = argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())

        start = best_path.pop()
        assert start == self.start_id
        best_path.reverse()

        scores = []
        if all_scores:
            for backscore in backscores:
                softmax = F.softmax(backscore, dim=0)
                scores.append([elem.item() for elem in softmax.flatten()])

            for index, (tag_id, tag_scores) in enumerate(zip(best_path, scores)):
                if type(tag_id) != int and tag_id.item() != np.argmax(tag_scores):
                    swap_index_score = np.argmax(tag_scores)
                    scores[index][tag_id.item()], scores[index][swap_index_score] = (
                        scores[index][swap_index_score],
                        scores[index][tag_id.item()],
                    )
                elif type(tag_id) == int and tag_id != np.argmax(tag_scores):
                    swap_index_score = np.argmax(tag_scores)
                    scores[index][tag_id], scores[index][swap_index_score] = (
                        scores[index][swap_index_score],
                        scores[index][tag_id],
                    )

        return best_scores, best_path, scores

    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
		Code from NCRFpp with some modification: https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py
        input:
            feats: (batch, seq_len, self.tag_size+2)
            mask: (batch, seq_len)
        output:
            decode_idx: (batch, nbest, seq_len) decoded sequence
            path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
            nbest decode for sentence with one token is not well supported, to be optimized
		"""

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert (tag_size == self.ntag)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        mask = mask.bool()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_id, :].clone()  # bat_size * to_target_size
        ## initial partition [batch_size, tag_size]
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        # iter over last scores
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(
                    batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            else:
                # previous to_target is current from_target
                # partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
                # cur_values: batch_size * from_target * to_target
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(
                    batch_size, tag_size, nbest, tag_size) + partition.contiguous().view(
                        batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
                ## compare all nbest and all from target
                cur_values = cur_values.view(batch_size, tag_size * nbest, tag_size)
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            if idx == 1:
                cur_bp = cur_bp * nbest
            partition = partition.transpose(2, 1)
            cur_bp = cur_bp.transpose(2, 1)

            #partition: (batch_size * to_target * nbest)
            #cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
            partition_history.append(partition)
            ## cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            ## mask[idx] ? mask[idx-1]
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0)
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, tag_size, nbest).transpose(
            1, 0).contiguous()  ## (batch_size, seq_len, nbest, tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size, 1, 1, 1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) \
            + self.transitions.view(1, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size * nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        ## end_partition: (batch, nbest, tag_size)
        end_bp = end_bp.transpose(2, 1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = torch.zeros(batch_size, tag_size, nbest, dtype=torch.long, device=feats.device)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        ## select end ids in STOP_TAG
        pointer = end_bp[:, self.stop_id, :]  ## (batch_size, nbest)
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        ## copy the ids of last position:insert_last to back_points, though the last_position index
        ## last_position includes the length of batch sentences
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        ## back_points: (seq_len, batch, tag_size, nbest)
        ## decode from the end, padded position ids are 0, which will be filtered in following evaluation
        decode_idx = torch.empty(seq_len, batch_size, nbest, dtype=torch.long, device=feats.device)

        decode_idx[-1] = pointer.data / nbest
        # use old mask, let 0 means has token
        for idx in range(len(back_points) - 2, -1, -1):
            new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size * nbest), 1,
                                       pointer.contiguous().view(batch_size, nbest))
            decode_idx[idx] = new_pointer.data / nbest
            # # use new pointer to remember the last end nbest ids for non longest
            pointer = new_pointer + pointer.contiguous().view(batch_size, nbest) * mask[idx].view(batch_size, 1).expand(
                batch_size, nbest).long()

        decode_idx = decode_idx.transpose(1, 0)

        ### calculate probability for each sequence
        scores = end_partition[:, :, self.stop_id]
        ## scores: [batch_size, nbest]
        max_scores, _ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size, 1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1)
        ## path_score: [batch_size, nbest]
        return path_score, decode_idx
