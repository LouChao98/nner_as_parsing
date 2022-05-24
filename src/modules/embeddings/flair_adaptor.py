try:
    import flair
    import torch
    from flair.data import Sentence
    from flair.embeddings import TokenEmbeddings
    from flair.training_utils import store_embeddings
    from src.my_typing import *

    from . import log
    from .adaptor import EmbeddingAdaptor


    class FlairEmbeddingAdaptor(EmbeddingAdaptor):
        def __init__(self, emb: TokenEmbeddings):
            super().__init__(emb)
            self._embed_size = self.emb.embedding_length
            log.info('You init a flair embedding, and you should make sure num_workers=0')

        @classmethod
        def trigger(cls, target_string):
            return target_string.startswith('flair')

        def process(self, vocabs, datasets):
            enable_flair_embedding(datasets)

        def forward(self, sentences: List[Sentence]):
            self.emb.embedding(sentences)
            lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
            longest_token_sequence_in_batch: int = max(lengths)

            pre_allocated_zero_tensor = torch.zeros(
                self.embed_size * longest_token_sequence_in_batch,
                device=self.device_indicator.device,
            )

            all_embs = []
            for sentence in sentences:
                all_embs += [emb for token in sentence for emb in token.get_each_embedding(self.emb.get_names())]
                nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

                if nb_padding_tokens > 0:
                    t = pre_allocated_zero_tensor[:self.emb.embedding_length * nb_padding_tokens]
                    all_embs.append(t)

            all_embs = torch.cat(all_embs).view(len(sentences), longest_token_sequence_in_batch, -1)
            store_embeddings(sentences, flair.embedding_storage_mode)
            return all_embs


    def enable_flair_embedding(datasets):
        class _Sentence(Sentence):
            def to(self, device: str = None, pin_memory: bool = False):
                if device is not None:
                    super().to(device, pin_memory)
                return self

        flair.embedding_storage_mode = 'none'
        for ds in datasets.values():
            ds.apply_field(_Sentence, 'raw_words', 'flair', is_input=True, ignore_type=True, padder=None)

except ImportError:
    FlairEmbeddingAdaptor = None
    enable_flair_embedding = None