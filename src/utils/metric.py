import logging

import torch
from fastNLP.core.vocabulary import Vocabulary
from torchmetrics import Metric

log = logging.getLogger('metric')


# All states will be stored on cpu.



class SpanMetric(Metric):
    def __init__(self, extra_vocab, unique, label_as_string, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique = unique
        self.label_as_string = label_as_string
        self.add_state('tp', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('utp', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('label', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('gold', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('pred', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('n', default=torch.tensor(0.), dist_reduce_fx='sum')

        self.span_label_vocab: Vocabulary = extra_vocab['span_label']
        self.convert = (lambda x: x) if label_as_string else (lambda x: self.span_label_vocab[x])

    def update(self, predict, gold, mask):
        # gold should not use set because duplicate spans are valid even they have the same boundary and label.
        pred_spans = [set([span for span in spans if span[2] != 0]) for spans in predict['span']]
        unlabeled_pred_spans = [set([(l, r) for (l, r, t) in spans]) for spans in predict['span']]
        pred_labels = [
            set([span for span in spans if span[2] != 0])
            for spans in predict.get('span_gold_boundary', [[]] * len(pred_spans))
        ]

        if self.unique:
            gold_spans = [
                set((l, r, self.convert(label)) for label, l, r in spans.values()) for spans in gold['raw_span']
            ]
            unlabeled_gold_spans = [set((left, right) for (left, right, label) in spans) for spans in gold_spans]
        else:
            gold_spans = [[(l, r, self.convert(label)) for label, l, r in spans.values()] for spans in gold['raw_span']]
            unlabeled_gold_spans = [[(left, right) for (left, right, label) in spans] for spans in gold_spans]
        assert len(pred_spans) == len(gold_spans)

        _tp, _utp, _label, _pred, _gold = 0, 0, 0, 0, 0
        for pred, gold, upred, ugold, label in zip(pred_spans, gold_spans, unlabeled_pred_spans, unlabeled_gold_spans,
                                                   pred_labels):
            if self.unique:
                _tp += len(gold.intersection(pred))
                _utp += len(ugold.intersection(upred))
                _label += len(gold.intersection(label))
            else:
                _tp += len([g for g in gold if g in pred])
                _utp += len([g for g in ugold if g in upred])
                _label += len([g for g in gold if g in label])
            _pred += len(pred)
            _gold += len(gold)

        self.tp += _tp
        self.utp += _utp
        self.label += _label
        self.pred += _pred
        self.gold += _gold
        self.n += len(pred_spans)

    def compute(self):
        log.debug(f'Gold span: {self.gold}, pred span: {self.pred}, total sent: {self.n}')
        return {
            'r': 100 * self.tp / (self.gold + 1e-12),
            'p': 100 * self.tp / (self.pred + 1e-12),
            'f1': 100 * 2 * self.tp / (self.pred + self.gold + 1e-12),
            'ur': 100 * self.utp / (self.gold + 1e-12),
            'lacc': 100 * self.label / (self.gold + 1e-12)
        }



class MultiMetric(Metric):
    def __init__(self, extra_vocab, **metric):
        super().__init__()
        self.metric = metric
        for name, metric in self.metric.items():
            self.add_module(name, metric)

    def update(self, predict, gold, mask):
        for m in self.metric.values():
            m.update(predict, gold, mask)

    def compute(self):
        out = {}
        for name, metric in self.metric.items():
            for key, value in metric.compute().items():
                if name == 'main':
                    out[key] = value
                else:
                    out[f'{name}/{key}'] = value
        return out

    def reset(self):
        for m in self.metric.values():
            m.reset()

    def __hash__(self) -> int:
        return hash(tuple(self.children()))


class SpanMetricDetail(MultiMetric):
    def __init__(self, extra_vocab, *args, **kwargs):
        super(SpanMetricDetail, self).__init__(extra_vocab)
        self.main = SpanMetric(extra_vocab, False, False, *args, **kwargs)
        self.uniq = SpanMetric(extra_vocab, True, False, *args, **kwargs)
        self.span1 = SpanMetric(extra_vocab, False, False, *args, **kwargs)
        self.span2p = SpanMetric(extra_vocab, False, False, *args, **kwargs)
        self.metric = {'main': self.main, 'uniq': self.uniq, 'span1': self.span1, 'span2p': self.span2p}

    def update(self, predict, gold, mask):
        self.main.update(predict, gold, mask)
        self.uniq.update(predict, gold, mask)

        predict1 = [[s for s in p if s[1] - s[0] == 1] for p in predict['span']]
        gold1 = [{k: v for k, v in g.items() if v[2] - v[1] == 1} for g in gold['raw_span']]
        self.span1.update({'span': predict1}, {'raw_span': gold1}, mask)
        predict2p = [[s for s in p if s[1] - s[0] != 1] for p in predict['span']]
        gold2p = [{k: v for k, v in g.items() if v[2] - v[1] != 1} for g in gold['raw_span']]
        self.span2p.update({'span': predict2p}, {'raw_span': gold2p}, mask)