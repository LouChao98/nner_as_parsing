import spacy

_nlp = None


def sentencizer_spacy(doc):
    # TODO: use sentencizer might cause lots of spans across sentences

    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

    n_newline = 0
    for i, _sent in enumerate(doc.split('\n')):
        prev = -n_newline
        sents = _nlp(_sent)
        sents.is_parsed = True  # spacy's bug
        for j, sent in enumerate(sents.sents):
            off = sent.start_char - prev
            prev = sent.end_char
            yield sent.text, off
            n_newline = len(_sent) - sent.end_char
        n_newline += 1  # for "\n"


def sentencizer_newline(doc):
    for i, _sent in enumerate(doc.split('\n')):
        yield _sent, 0 if i == 0 else 1


def sentencizer_concat(doc: str):
    yield doc.replace('\n', ' '), 0


sentencizers = {
    'spacy': sentencizer_spacy,
    'newline': sentencizer_newline,
    'concat': sentencizer_concat
}
