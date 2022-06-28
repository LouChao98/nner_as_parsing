#!/usr/bin/env python

# Based on https://github.com/neulab/cmu-multinlp/blob/master/data/conllXtostandoff.py

# Script to convert a CoNLL X (2006) tabbed dependency tree format
# file into BioNLP ST-flavored standoff and a reconstruction of the
# original text.

import codecs
import os
import re
import sys

# maximum number of sentences to include in single output document
# (if None, doesn't split into documents)
MAX_DOC_SENTENCES = [10]

# whether to output an explicit root note
OUTPUT_ROOT = False
# the string to use to represent the root node
ROOT_STR = 'ROOT'

INPUT_ENCODING = 'UTF-8'
OUTPUT_ENCODING = 'UTF-8'

output_directory = None

# rewrites for characters appearing in CoNLL-X types that cannot be
# directly used in identifiers in brat-flavored standoff
charmap = {
    '<': '_lt_',
    '>': '_gt_',
    '+': '_plus_',
    '?': '_question_',
    '&': '_amp_',
    ':': '_colon_',
    '.': '_period_',
    '!': '_exclamation_',
}


def maptype(s):
    return ''.join([charmap.get(c, c) for c in s])


def tokstr(start, end, ttype, idnum, text):
    # sanity checks
    assert '\n' not in text, "ERROR: newline in entity '%s'" % (text)
    assert text == text.strip(), "ERROR: tagged span contains extra whitespace: '%s'" % (text)
    return 'T%d\t%s %d %d\t%s' % (idnum, maptype(ttype), start, end, text)


def depstr(depid, headid, rel, idnum):
    return 'R%d\t%s Arg1:T%d Arg2:T%d' % (idnum, maptype(rel), headid, depid)


def output(infn, docnum, sentences):
    global output_directory

    if output_directory is None:
        txtout = sys.stdout
        soout = sys.stdout
    else:
        # add doc numbering if there is a sentence count limit,
        # implying multiple outputs per input
        if MAX_DOC_SENTENCES:
            outfnbase = os.path.basename(infn).rsplit('.', 1)[0] + '-doc-' + str(docnum)
        else:
            outfnbase = os.path.basename(infn).rsplit('.', 1)[0]
        outfn = os.path.join(output_directory, outfnbase)
        txtout = codecs.open(outfn + '.txt', 'w', encoding=OUTPUT_ENCODING)
        soout = codecs.open(outfn + '.ann', 'w', encoding=OUTPUT_ENCODING)

    offset, idnum, ridnum = 0, 1, 1

    doctext = '\n'.join(' '.join(tokens) for tokens, _ in sentences)
    print(doctext, file=txtout)

    for si, sentence in enumerate(sentences):
        tokens, spans = sentence

        offset_one_sent = [offset]

        for token in tokens:
            offset_one_sent.append(offset_one_sent[-1] + len(token) + 1)
        offset = offset_one_sent[-1]
        for span in spans:
            boundary, label = span.split(' ')
            l, r = list(map(int, boundary.split(',')))

            # output a token annotation
            print(tokstr(offset_one_sent[l], offset_one_sent[r]-1, label, idnum, ' '.join(tokens[l:r])), file=soout)
            idnum += 1
            if doctext[offset_one_sent[l]:offset_one_sent[r] - 1] != ' '.join(tokens[l:r]):
                breakpoint()

        # offset += 1



def process(fn):
    docnum = 1
    sentences = []

    with codecs.open(fn, encoding=INPUT_ENCODING) as f:

        tokens, deps = [], []

        lines = f.readlines()
        output_entity = 0
        output_entity_unique = 0

        for ln in range(len(lines)):
            if ln % 3:
                continue

            tokens = lines[ln].strip().split()
            spans = lines[ln + 1].strip()

            if spans:
                spans = spans.split('|')
            else:
                spans = []

            sentences.append((tokens, spans))

            # limit sentences per output "document"
            if MAX_DOC_SENTENCES and len(sentences) >= MAX_DOC_SENTENCES[0]:
                output(fn, docnum, sentences)
                output_entity += sum(len(span) for _, span in sentences)
                for sent in sentences:
                    output_entity_unique += len(set(sent[1]))
                sentences = []
                docnum += 1
    if len(sentences):
        output(fn, docnum, sentences)
        output_entity += sum(len(span) for _, span in sentences)
    print('output entity', output_entity)
    print('output output_entity_unique', output_entity_unique)


def main(argv):
    global output_directory

    # Take an optional "-o" arg specifying an output directory for the results
    output_directory = None
    filenames = argv[1:]
    if len(argv) > 2 and argv[1] == '-o':
        output_directory = argv[2]
        print('Writing output to %s' % output_directory, file=sys.stderr)
        filenames = argv[3:]

    if argv[-1].startswith('sent:'):
        MAX_DOC_SENTENCES[0] = int(argv[-1][5:])
        filenames = filenames[:-1]
        print('MAX_DOC_SENTENCES {}'.format(MAX_DOC_SENTENCES))

    fail_count = 0
    for fn in filenames:
        process(fn)
        # try:
        # except Exception as e:
        #     m = str(e).encode(OUTPUT_ENCODING)
        #     print('Error processing %s: %s' % (fn, m), file=sys.stderr)
        #     fail_count += 1

    if fail_count > 0:
        print("""
##############################################################################
#
# WARNING: error in processing %d/%d files, output is incomplete!
#
##############################################################################
""" % (fail_count, len(filenames)),
              file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
