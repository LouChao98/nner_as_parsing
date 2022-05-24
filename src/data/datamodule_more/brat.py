import operator
import os
from collections import defaultdict
from functools import reduce

import numpy as np
import regex
import torch
from fastNLP import Instance
from fastNLP.core.field import Padder, SentFeat2DCatPadder, SentFeat2DPadder
from fastNLP.io.loader.loader import Loader
from nltk.corpus import stopwords
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tree import Tree
from sparse import COO
from torch.nn.functional import pad

# noinspection PyUnresolvedReferences
from src.data.datamodule import log, DataModule
from src.data.datamodule_more._sentencizer import sentencizers
from src.my_typing import *

# Based on https://github.com/neulab/cmu-multinlp

class MyToken:
    def __init__(self, text, idx):
        self.text = text
        self.idx = idx

    def __str__(self):
        return str((self.text, self.idx))

    def __repr__(self):
        return str((self.text, self.idx))


def tokenizer_space(sent):
    tokens = []
    offset = 0
    for i, t in enumerate(sent.split(' ')):
        tokens.append(MyToken(t, offset))
        offset += len(t) + 1  # for space
    return tokens


def strip_functional_tags(tree: Tree) -> None:
    """
    Removes all functional tags from constituency labels in an NLTK tree.
    We also strip off anything after a =, - or | character, because these
    are functional tags which we don't want to use.
    This modification is done in-place.
    """
    clean_label = tree.label().split('=')[0].split('-')[0].split('|')[0]
    tree.set_label(clean_label)
    for child in tree:
        if not isinstance(child[0], str):
            strip_functional_tags(child)


def get_trees_from_bracket_file(filename) -> List[Tree]:
    directory, filename = os.path.split(filename)
    trees = list(BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents())
    modified_trees = []
    for tree in trees:
        strip_functional_tags(tree)
        # This is un-needed and clutters the label space.
        # All the trees also contain a root S node.
        if tree.label() == 'VROOT' or tree.label() == 'TOP':
            tree = tree[0]
        modified_trees.append(tree)
    return modified_trees


def adjust_tokens_wrt_char_boundary(tokens: List[MyToken], char_boundaries: List[int]):
    """
    Positions indicated by char_boundaries should be segmented.
    If one of the indices is 3, it mean that there is a boundary between the 3rd and 4th char.
    Indices in char_boundaries should be in ascending order.
    """
    new_tokens: List[MyToken] = []
    cb_ind = 0
    for tok in tokens:
        start = tok.idx
        end = tok.idx + len(tok.text)
        ext_bd = []
        while cb_ind < len(char_boundaries) and char_boundaries[cb_ind] <= end:
            bd = char_boundaries[cb_ind]
            if bd != start and bd != end:  # boundary not detected by tokenizer
                ext_bd.append(bd)
            cb_ind += 1
        for s, e in zip([start] + ext_bd, ext_bd + [end]):
            text = tok.text[s - start:e - start]
            new_tokens.append(MyToken(text, s))
    return new_tokens


def get_boundary(span):
    # span = (label, start, end)
    # start end can be list if discontinuous (no overlap)
    # sort by start
    if isinstance(span[1], int):
        return span[1], span[2]
    else:
        return span[1][0], span[2][-1]


class BratDoc:
    EVENT_JOIN_SYM = '->'

    def __init__(
            self,
            id: str,
            doc: Union[str, List[str]],
            # span_id -> (span_label, start_ind, end_ind),
            # where start_ind is inclusive and end_ind is exclusive
            spans: Dict[str, Tuple[str, int, int]],
            # (span_id1, span_id2) -> span_pair_label
            span_pairs: Dict[Tuple[str, str], str],
            bracket_file: str = None,
            tree: Tree = None,
            cluster=None):
        self.id = id
        self.doc = doc  # can be str of chars or a list of tokens
        self.spans = spans
        self.span_pairs = span_pairs
        self.bracket_file = bracket_file
        self.tree = tree
        self.cluster = cluster

    def get_span_weights(self) -> Dict[str, float]:
        """ compute the weight of the span by how many times it appears in span pairs """
        span2count: Dict[str, int] = defaultdict(lambda: 0)
        for sid1, sid2 in self.span_pairs:
            span2count[sid1] += 1
            span2count[sid2] += 1
        return dict((k, float(span2count[k])) for k in self.spans)

    def remove_span_pair_not_in_label_set(self, labels: set):
        self.span_pairs = dict((k, v) for k, v in self.span_pairs.items() if v not in labels)

    def remove_span_not_in_label_set(self, labels: set):
        self.spans = dict((k, v) for k, v in self.spans.items() if v[0] not in labels)

    def remove_span_not_in_pair(self):
        sp_set = set(k_ for k in self.span_pairs for k_ in k)
        self.spans = dict((k, v) for k, v in self.spans.items() if k in sp_set)

    def build_cluster(self, inclusive=False) -> List[List[Tuple[int, int]]]:
        cluster: Dict[Tuple[int, int], int] = {}
        num_clusters = 0
        num_overlap_pairs = 0
        for k1, k2 in self.span_pairs:
            offset = 1 if inclusive else 0
            span_parent = (self.spans[k1][1], self.spans[k1][2] - offset)
            span_child = (self.spans[k2][1], self.spans[k2][2] - offset)
            if self.spans[k1][1] < self.spans[k2][2]:
                num_overlap_pairs += 1
            if span_child not in cluster and span_parent not in cluster:
                cluster[span_child] = num_clusters
                cluster[span_parent] = num_clusters
                num_clusters += 1
            elif span_child in cluster and span_parent in cluster:
                if cluster[span_parent] != cluster[span_child]:  # merge
                    from_clu = cluster[span_parent]
                    to_clu = cluster[span_child]
                    for k in cluster:
                        if cluster[k] == from_clu:
                            cluster[k] = to_clu
            elif span_child in cluster:
                cluster[span_parent] = cluster[span_child]
            elif span_parent in cluster:
                cluster[span_child] = cluster[span_parent]
        result = defaultdict(list)
        for k, v in cluster.items():
            result[v].append(k)
        self.cluster = list(result.values())
        return self.cluster

    def to_word(self):
        """ segment doc and convert char-based index to word-based index """
        # tokenize
        toks = tokenizer_space(self.doc)
        char_bd = set()
        for sid, (slabel, start, end) in self.spans.items():
            if isinstance(start, int):
                start, end = [start], [end]
            for s, e in zip(start, end):
                char_bd.add(s)
                char_bd.add(e)
        toks = adjust_tokens_wrt_char_boundary(toks, char_boundaries=sorted(char_bd))
        words = [tok.text for tok in toks]
        # build char ind to token ind mapping
        idxs = [(tok.idx, tok.idx + len(tok.text)) for tok in toks]
        sidx2tidx = dict((s[0], i) for i, s in enumerate(idxs))  # char start ind -> token ind
        eidx2tidx = dict((s[1], i) for i, s in enumerate(idxs))  # char end ind -> token ind
        # convert spans
        new_spans = {}
        for sid, (span_label, sidx, eidx) in self.spans.items():
            if isinstance(sidx, int):
                sidx, eidx = [sidx], [eidx]
            new_boundary = []
            for s, e in zip(sidx, eidx):
                if s in sidx2tidx and e in eidx2tidx:
                    new_boundary.append((sidx2tidx[s], eidx2tidx[e] + 1))  # end index is exclusive
                else:  # remove blanks and re-check
                    span_str = self.doc[s:e]
                    blank_str = len(span_str) - len(span_str.lstrip())
                    blank_end = len(span_str) - len(span_str.rstrip())
                    s += blank_str
                    e -= blank_end
                    if s in sidx2tidx and e in eidx2tidx:
                        new_boundary.append((sidx2tidx[s], eidx2tidx[e] + 1))  # end index is exclusive
                    else:  # the annotation boundary is not consistent with the tokenization boundary
                        raise Exception
            new_boundary.sort(key=lambda x: x[0])
            new_spans[sid] = (span_label, *(new_boundary[0] if len(new_boundary) == 1 else zip(*new_boundary)))

        # convert span pairs
        new_span_pairs = dict(
            ((s1, s2), v) for (s1, s2), v in self.span_pairs.items() if s1 in new_spans and s2 in new_spans)
        return BratDoc(self.id, words, new_spans, new_span_pairs, bracket_file=self.bracket_file, tree=self.tree)

    def split_by_sentence(self, sentencizer=None) -> List['BratDoc']:
        """ split into multiple docs by sentence boundary """
        sents = list(sentencizer(self.doc))  # sentencizer should return the offset between two adjacent sentences

        # split bracket file
        if self.bracket_file:
            trees = get_trees_from_bracket_file(self.bracket_file)
            assert len(trees) == len(sents), '#sent not equal to #tree'

        # collect spans for each sentence
        spans_ord = sorted(self.spans.items(), key=lambda x: get_boundary(x[1]))  # sorted by start ind and end ind
        num_skip_char = 0
        span_ind = 0
        spans_per_sent = []
        for i, (sent, off) in enumerate(sents):
            num_skip_char += off
            spans_per_sent.append([])
            cur_span = spans_per_sent[-1]
            # start ind and end ind should be not larger than a threshold
            while span_ind < len(spans_ord):
                start, end = get_boundary(spans_ord[span_ind][1])
                if start >= num_skip_char + len(sent) and end > num_skip_char + len(sent):
                    break
                if start < num_skip_char or end <= num_skip_char:
                    log.warning('span is spreaded across sentences')
                    span_ind += 1
                    continue
                sid, (slabel, sind, eind) = spans_ord[span_ind]
                if isinstance(sind, int):
                    cur_span.append((sid, (slabel, sind - num_skip_char, eind - num_skip_char)))
                else:
                    cur_span.append(
                        (sid, (slabel, [s - num_skip_char for s in sind], [e - num_skip_char for e in eind])))
                span_ind += 1
            num_skip_char += len(sent)

        # collect span pairs for each sentence
        pair_count = 0
        brat_doc_li = []
        for i, spans in enumerate(spans_per_sent):
            if len(sents[i][0]) <= 0:  # skip empty sentences
                continue
            span_ids = set(span[0] for span in spans)
            span_pair = dict(
                ((s1, s2), v) for (s1, s2), v in self.span_pairs.items() if s1 in span_ids and s2 in span_ids)
            pair_count += len(span_pair)
            tree = trees[i] if self.bracket_file else None
            brat_doc_li.append(BratDoc(self.id, sents[i][0], dict(spans), span_pair, tree=tree))
        # TODO: span pairs across sentences are allowed
        # assert pair_count == len(self.span_pairs), 'some span pairs are spreaded across sentences'

        return brat_doc_li

    @classmethod
    def dummy(cls):
        return cls('id', ['token'], {}, {})

    @classmethod
    def from_file(cls, text_file: str, ann_file: str, bracket_file: str = None):
        """ read text and annotations from files in BRAT format """
        with open(text_file, 'r') as txtf:
            doc = txtf.read().rstrip()
        spans = {}
        span_pairs = {}
        eventid2triggerid = {}  # e.g., E10 -> T27
        with open(ann_file, 'r') as annf:
            for l in annf:
                if l.startswith('#'):  # skip comment
                    continue
                if l.startswith('T'):
                    # 1. there are some special chars at the end of the line, so we only strip \n
                    # 2. there are \t in text spans, so we only split twice
                    ann = l.rstrip('\t\n').split('\t', 2)
                else:
                    ann = l.rstrip().split('\t')
                aid = ann[0]
                if aid.startswith('T'):  # text span annotation
                    span_label, boundary = ann[1].split(' ', maxsplit=1)
                    boundary = [list(map(int, one.split())) for one in boundary.split(';')]
                    if len(boundary) == 1:
                        sind, eind = boundary[0]
                        spans[aid] = (span_label, sind, eind)
                        # sanity check, some times doc[sind:eind] contains space, I remove them in this check
                        # because to_word will remove them anyway.
                        if len(ann) > 2:
                            assert ann[2].strip() == doc[sind:eind].strip(
                            ), f"expect '{ann[2]}', but '{doc[sind:eind]}'"
                    else:
                        spans[aid] = (span_label, *zip(*boundary))
                elif aid.startswith('E'):  # event span annotation
                    events = ann[1].split(' ')
                    trigger_type, trigger_aid = events[0].split(':')
                    eventid2triggerid[aid] = trigger_aid
                    for event in events[1:]:
                        arg_type, arg_aid = event.split(':')
                        span_pairs[(trigger_aid, arg_aid)] = trigger_type + cls.EVENT_JOIN_SYM + arg_type
                elif aid.startswith('R'):  # relation annotation
                    rel = ann[1].split(' ')
                    assert len(rel) == 3
                    rel_type = rel[0]
                    arg1_aid = rel[1].split(':')[1]
                    arg2_aid = rel[2].split(':')[1]
                    if (arg1_aid, arg2_aid) in span_pairs:
                        log.warning(f'{ann} has duplicate span pairs, '
                                    f'previous is {span_pairs[(arg1_aid, arg2_aid)]}, now is {rel_type}')
                    span_pairs[(arg1_aid, arg2_aid)] = rel_type
                elif aid.startswith('N'):  # normalization annotation
                    # TODO: how to deal with normalization?
                    log.warning(f'Unhandled normalization annotation: {l}')
                elif not aid[0].istitle():  # skip lines not starting with upper case characters
                    log.warning(f'Skipping line due to istitle=False: {l}')
                else:
                    raise NotImplementedError

        # convert event id to text span id
        span_pairs_converted = {}
        for (sid1, sid2), v in span_pairs.items():
            if sid1.startswith('E'):
                sid1 = eventid2triggerid[sid1]
            if sid2.startswith('E'):
                sid2 = eventid2triggerid[sid2]
            span_pairs_converted[(sid1, sid2)] = v
        return cls(ann_file, doc, spans, span_pairs_converted, bracket_file=bracket_file)


class _BratLoader(Loader):
    def __init__(self, sentencizer):
        super().__init__()
        self.sentencizer = sentencizers[sentencizer]

    def download(self) -> str:
        pass

    def _load(self, root_dir):
        ds = DataSet()
        for brat_doc in self.walk(root_dir):
            brat_doc = brat_doc.to_word()
            inst = {'raw_words': brat_doc.doc, 'raw_span': brat_doc.spans, 'raw_span_pair': brat_doc.span_pairs}
            if brat_doc.tree:
                inst['tree'] = brat_doc.tree
            ds.append(Instance(**inst))

        if len(ds) == 0:
            raise ValueError(f'Can not find files in the given path: {root_dir}')

        ds['raw_span'].ignore_type = True
        ds['raw_span_pair'].ignore_type = True

        if 'tree' in ds:
            ds['tree'].ignore_type = True

        return ds

    def walk(self, root_dir):
        for root, dirs, files in os.walk(root_dir, followlinks=True):
            for file in files:
                if not file.endswith('.txt'):
                    continue
                text_file = os.path.join(root, file)
                ann_file = os.path.join(root, file[:-4] + '.ann')
                bracket_file = os.path.join(root, file[:-4] + '.bracket')
                if not os.path.exists(ann_file):
                    log.warning(f'No matching .ann file for {text_file}')
                    continue
                if not os.path.exists(bracket_file):
                    bracket_file = None
                brat_doc = BratDoc.from_file(text_file, ann_file, bracket_file=bracket_file)
                yield from brat_doc.split_by_sentence(sentencizer=self.sentencizer)


class BratNER(DataModule):
    INPUTS = ('id', 'words', 'seq_len')
    TARGETS = ('span_label', 'span_mask', 'raw_span', 'num_span', 'num_span2')
    EXTRA_VOCAB = ('span_label', )
    LOADER = _BratLoader

    def __init__(
            self,
            sentencizer: str = 'newline',
            mask_stopword=False,
            mask_punct=False,
            num_label=None,
            train_unk=False,
            span_indicator=False,
            span_sub_indicator=False,
            span_list=False,
            suffix=None,
            forbid_shared_root=False,  # if true, nested ner must have different root
            allow_empty=False,
            **kwargs):
        suffix = suffix or []
        self.mask_stopword = mask_stopword
        if self.mask_stopword:
            self.stopword_set = set(stopwords.words('english'))
            self.TARGETS = self.TARGETS + ('arc_mask', )
            suffix.append('stop')
        self.mask_punct = mask_punct
        if self.mask_punct:
            self.TARGETS = self.TARGETS + ('arc_mask', )
            suffix.append('mpunct')
        self.num_label = num_label  # if set, will produce span_label with shape [n x n x L]. 1 for observed.
        if self.num_label is not None:
            suffix.append('ml')
        self.train_unk = train_unk
        if self.train_unk:
            assert self.num_label is not None
            suffix.append('tu')
        self.span_indicator = span_indicator
        self.span_sub_indicator = span_sub_indicator
        if self.span_indicator:
            self.TARGETS = self.TARGETS + ('span_indicator', )
            if self.span_sub_indicator:
                suffix.append('sbi')
            else:
                suffix.append('si')
        self.span_list = span_list
        if self.span_list:
            self.TARGETS = self.TARGETS + ('span_list', )
            suffix.append('sll')
        self.forbid_shared_root = forbid_shared_root
        if self.forbid_shared_root:
            suffix.append('fsr')
        self.allow_empty = allow_empty
        if self.allow_empty:
            suffix.append('empty')

        super().__init__(self.LOADER(sentencizer), suffix=suffix, **kwargs)

    def _load(self, path, is_eval, set_target):
        ds: DataSet = self.loader._load(path)
        # each inst has n = seq_len + 1
        #   text
        #   span_label          [n x n], 0 for neg (is not a span)
        #   span_mask           [n x n], True for masked (can not exists)
        #   arc_mask            [n x n], True for masked (can not exists)
        #   --------------
        #   span_list           list of tuple of [i, j) + label

        ds.apply_more(self.gen)
        if not self.allow_empty and not is_eval:
            len_orig = len(ds)
            ds.drop(lambda x: x['empty'])
            len_non_empty = len(ds)
            log.warning(f'Dropping {len_orig - len_non_empty} empty instances.')
        else:
            len_non_empty = len(ds)

        cleaned_ds = ds.drop(lambda x: not x['valid'], inplace=not is_eval)
        len_valid = len(cleaned_ds)

        if len_non_empty != len_valid:
            if not is_eval:
                log.warning(f'Dropping {len_non_empty - len_valid} invalid instances.')
            else:
                log.warning(f'{len_non_empty - len_valid} instances is invalid for the model.')

        if self.num_label is not None:
            ds.set_padder('span_label', MultiSpanLabelPadder(self.num_label, self.train_unk))
            ds.set_ignore_type('span_label')
        else:
            ds.set_padder('span_label', SentFeat2DPadder(0))  # vocab should guarantee <unk> is 0

        ds.set_padder('span_mask', SentFeat2DPadder(False))
        ds.add_seq_len('raw_span', 'num_span')
        ds.apply_field(lambda spans: len([l for _, l, r in spans.values() if r - l > 1]), 'raw_span', 'num_span2')

        if self.mask_stopword:
            ds.apply_more(self.update_mask_using_stopword)
            ds.set_padder('arc_mask', SentFeat2DPadder(False))

        if self.mask_punct:
            ds.apply_more(self.update_mask_using_punct)
            ds.set_padder('arc_mask', SentFeat2DPadder(False))

        if self.span_indicator:
            ds.apply_more(self.gen_span_indicator)
            ds.set_padder('span_indicator', SentFeat2DPadder(False))

        if self.span_list:
            ds.apply_more(self.gen_span_list)
            ds.set_padder('span_list', SpanListPadder())
            ds.set_ignore_type('span_list')

        if self.forbid_shared_root:
            ds.apply_more(self.update_mask_forbid_shared_root)
            ds.set_padder('arc_mask', SentFeat2DPadder(False))

        return ds

    def gen(self, inst):
        seq_len = len(inst['raw_words'])
        raw_spans: Dict[str, Tuple[str, int, int]] = inst['raw_span']

        if self.num_label is not None:
            span_label = [[[] for _ in range(seq_len + 1)] for _ in range(seq_len + 1)]
        else:
            span_label = np.full((seq_len + 1, seq_len + 1), fill_value='<unk>', dtype=object)
        span_mask = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)
        warned = False
        for label, left, right in raw_spans.values():
            if self.num_label is not None:
                span_label[left][right].append(label)
            else:
                if span_label[left, right] != '<unk>' and span_label[left, right] != label and not warned:
                    log.warning(f'Overwriting label: {" ".join(inst["raw_words"])}')
                    warned = True
                span_label[left, right] = label
            span_mask[:left, left + 1:right] = True
            span_mask[left + 1:right, right + 1:] = True
        # sanity check: no crossing
        valid = True
        for label, left, right in raw_spans.values():
            if span_mask[left, right]:
                valid = False
                break
        if not valid:
            span_mask.fill(True)  # I can not find a better way to forbid access to span_mask if not valid.
        return {'span_label': span_label, 'span_mask': span_mask, 'valid': valid, 'empty': len(raw_spans) == 0}

    def gen_span_indicator(self, inst):
        seq_len = len(inst['raw_words'])
        raw_spans: Dict[str, Tuple[str, int, int]] = inst['raw_span']
        span_indicator = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)
        for label, left, right in raw_spans.values():
            span_indicator[left, right] = True
            if self.span_sub_indicator:
                span_indicator[left:right, left + 1:right + 1] = \
                    np.triu(np.ones((right - left, right - left), dtype=np.bool8))

        return {'span_indicator': span_indicator}

    def gen_span_list(self, inst):
        raw_spans: Dict[str, Tuple[str, int, int]] = inst['raw_span']
        span_dict = defaultdict(list)
        for label, left, right in raw_spans.values():
            span_dict[(left, right)].append(label)
        return {
            'span_list': [[left, right] for left, right in span_dict.keys()],
        }

    def update_mask_using_stopword(self, inst):
        seq_len = len(inst['raw_words'])
        if 'arc_mask' in inst.dataset:
            arc_mask = inst['arc_mask']
        else:
            arc_mask = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)

        is_stopwords = [False for _ in range(seq_len)]
        words = [w.lower() for w in inst['raw_words']]
        for i, w in enumerate(words):
            if w in self.stopword_set:
                is_stopwords[i] = True

        for _, l, r in inst['raw_span'].values():
            if all(is_stopwords[l:r]):
                is_stopwords[l:r] = [False] * (r - l)

        for i, f in enumerate(is_stopwords, start=1):
            if f:
                arc_mask[i] = True
        return {'arc_mask': arc_mask}

    def update_mask_using_punct(self, inst):
        seq_len = len(inst['raw_words'])
        if 'arc_mask' in inst.dataset:
            arc_mask = inst['arc_mask']
        else:
            arc_mask = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)

        is_punct = [False for _ in range(seq_len)]
        words = [w.lower() for w in inst['raw_words']]
        for i, w in enumerate(words):
            if regex.match(r'\p{P}+$', w):
                is_punct[i] = True

        for _, l, r in inst['raw_span'].values():
            if all(is_punct[l:r]):
                is_punct[l:r] = [False] * (r - l)

        for i, f in enumerate(is_punct, start=1):
            if f:
                arc_mask[i] = True
        return {'arc_mask': arc_mask}

    def update_mask_forbid_shared_root(self, inst):
        seq_len = len(inst['raw_words'])
        if 'arc_mask' in inst.dataset:
            arc_mask = inst['arc_mask']
        else:
            arc_mask = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)

        span = [(l, r) for _, l, r in inst['raw_span'].values()]
        direct_upper: List[Optional[Tuple[int, int]]] = [None for _ in range(len(span))]
        for i, s1 in enumerate(span):
            for s2 in span:
                if s1 == s2: continue
                if s2[0] <= s1[0] and s1[1] <= s2[1]:
                    if direct_upper[i] is None:
                        direct_upper[i] = s2
                    else:
                        s3 = direct_upper[i]
                        if s3[0] <= s2[0] and s2[1] <= s3[1]:
                            direct_upper[i] = s2

        for s, u in zip(span, direct_upper):
            if u is None: continue
            arc_mask[:, s[0] + 1:s[1] + 1] = True
            arc_mask[u[0] + 1:u[1] + 1, s[0] + 1:s[1] + 1] = False
        return {'arc_mask': arc_mask}

    def get_bos_eos_adder(self, field):
        if field in ('span_label', 'span_mask', 'span_indicator', 'arc_mask', 'raw_span', 'num_span', 'num_span2',
                     'span_list'):
            return None
        return super().get_bos_eos_adder(field)


class SpanListPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_num_span = max(len(c) for c in contents)
        padded_array = np.full((len(contents), max_num_span, 2), fill_value=0, dtype=np.int_)
        for b_idx, c in enumerate(contents):
            if len(c) > 0:
                padded_array[b_idx, :len(c)] = np.asarray(c)
        return torch.tensor(padded_array)


class MultiSpanLabelPadder(Padder):
    def __init__(self, num_tag, train_unk, pad_val=0, **kwargs):
        super().__init__(pad_val=pad_val, **kwargs)
        self.num_tag = num_tag
        self.train_unk = train_unk

    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        batch_size, max_len, = len(contents), max(len(c) for c in contents)
        padded_array = np.full((batch_size, max_len, max_len, self.num_tag), fill_value=0, dtype=np.int_)
        for b_idx, matrix in enumerate(contents):
            for r_idx, row in enumerate(matrix):
                for c_idx, column in enumerate(row):
                    for i in column:
                        padded_array[b_idx, r_idx, c_idx, i] = 1.
                    if self.train_unk and len(column) == 0:
                        padded_array[b_idx, r_idx, c_idx, 0] = 1.
        return torch.tensor(padded_array)


class DiscontinuousBratNER(DataModule):
    INPUTS = ('id', 'words', 'seq_len')
    TARGETS = ('span_label', 'span_mask', 'raw_span', 'num_span', 'num_span2')
    EXTRA_VOCAB = ('span_label', )
    LOADER = _BratLoader

    # no arc

    def __init__(self,
                 sentencizer: str = 'newline',
                 mask_stopword=False,
                 num_label=None,
                 span_indicator=False,
                 suffix=None,
                 **kwargs):
        self.mask_stopword = mask_stopword
        suffix = suffix or []
        if self.mask_stopword:
            self.stopword_set = set(stopwords.words('english'))
            self.TARGETS = self.TARGETS + ('arc_mask', )
            suffix.append('stop')
        self.num_label = num_label  # if set, will produce span_label with shape [n x n x L]. 1 for observed.
        if self.num_label is not None:
            suffix.append('ml')  # TODO
        self.span_indicator = span_indicator
        if self.span_indicator:
            self.TARGETS = self.TARGETS + ('span_indicator', )
            suffix.append('si')
        super().__init__(self.LOADER(sentencizer), suffix=suffix, **kwargs)

    def _load(self, path, is_eval, set_target):
        ds: DataSet = self.loader._load(path)
        # each inst has n = seq_len + 1
        #   text
        #   span_label          [n x n], 0 for neg (is not a span)
        #   span_mask           [n x n], True for masked (can not exists)
        #   arc_mask            [n x n], True for masked (can not exists)
        #   --------------
        #   span_list           list of tuple of [i, j) + label

        ds.apply_more(self.gen)

        raise NotImplementedError
        len_orig = len(ds)  # TODO do not drop dev/test
        ds.drop(lambda x: x['empty'])
        len_non_empty = len(ds)
        log.warning(f'Dropping {len_orig - len_non_empty} empty instances.')

        cleaned_ds = ds.drop(lambda x: not x['valid'], inplace=not is_eval)
        len_valid = len(cleaned_ds)

        if len_non_empty != len_valid:
            if not is_eval:
                log.warning(f'Dropping {len_non_empty - len_valid} invalid instances.')
            else:
                log.warning(f'{len_non_empty - len_valid} instances is invalid for the model.')

        ds.set_padder('span_label', SentFeat2DPadder(0))  # vocab should guarantee <unk> is 0
        ds.set_padder('span_mask', SentFeat2DPadder(False))
        ds.add_seq_len('raw_span', 'num_span')
        ds.apply_field(lambda spans: len([l for _, l, r in spans.values() if r - l > 1]), 'raw_span', 'num_span2')

        if self.mask_stopword:
            ds.apply_more(self.update_mask_using_stopword)
            ds.set_padder('arc_mask', SentFeat2DPadder(False))

        if self.span_indicator:
            ds.apply_more(self.gen_span_indicator)
            ds.set_padder('span_indicator', SentFeat2DPadder(False))

        return ds

    def gen(self, inst):
        seq_len = len(inst['raw_words'])
        raw_spans: Dict[str, Tuple[str, int, int]] = inst['raw_span']

        span_label = np.full((seq_len + 1, seq_len + 1), fill_value='<unk>', dtype=object)
        span_mask = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)
        warned = False
        for label, left, right in raw_spans.values():
            if span_label[left, right] != '<unk>' and not warned:
                log.warning(f'Overwriting label: {" ".join(inst["raw_words"])}')
                warned = True
            # assert span_label[left, right] in ('<unk>', label)
            span_label[left, right] = label
            span_mask[:left, left + 1:right] = True
            span_mask[left + 1:right, right + 1:] = True
        # sanity check: no crossing
        valid = True
        for label, left, right in raw_spans.values():
            if span_mask[left, right]:
                valid = False
                break
        if not valid:
            span_mask.fill(True)  # I can not find a better way to forbid access to span_mask if not valid.
        return {'span_label': span_label, 'span_mask': span_mask, 'valid': valid, 'empty': len(raw_spans) == 0}

    def gen_span_indicator(self, inst):
        seq_len = len(inst['raw_words'])
        raw_spans: Dict[str, Tuple[str, int, int]] = inst['raw_span']
        span_indicator = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)
        for label, left, right in raw_spans.values():
            span_indicator[left, right] = True
        return {'span_indicator': span_indicator}

    def update_mask_using_stopword(self, inst):
        seq_len = len(inst['raw_words'])
        arc_mask = np.zeros((seq_len + 1, seq_len + 1), dtype=np.bool8)

        words = [w.lower() for w in inst['raw_words']]
        for i, w in enumerate(words, start=1):
            if w in self.stopword_set:
                arc_mask[i] = True
        return {'arc_mask': arc_mask}

    def get_bos_eos_adder(self, field):
        if field in ('span_label', 'span_mask', 'span_indicator', 'arc_mask', 'raw_span', 'num_span', 'num_span2'):
            return None
        return super().get_bos_eos_adder(field)


class ConcatPadder(Padder):
    def __init__(self, pad_val=0, **kwargs):
        super().__init__(pad_val=pad_val, **kwargs)

    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        return reduce(operator.add, contents)

