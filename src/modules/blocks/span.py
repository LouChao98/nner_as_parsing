# from https://github.com/foxlf823/sodner
# from https://github.com/neulab/cmu-multinlp

from math import floor

import torch
import torch.nn as nn
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor
from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import util
from overrides import overrides
from torch.nn import Conv1d
from torch.nn.init import xavier_normal_


class EndpointSpanExtractor(nn.Module):
    """
    Represents spans as a combination of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.

    The following types of representation are supported, assuming that
    ``x = span_start_embeddings`` and ``y = span_end_embeddings``.

    ``x``, ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give ``x,y,x*y`` as the ``combination`` parameter to this class.
    The computed similarity function would then be ``[x; y; x*y]``, which can then be optionally
    concatenated with an embedded representation of the width of the span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.
    combination : str, optional (default = "x,y").
        The method used to combine the ``start_embedding`` and ``end_embedding``
        representations. See above for a full description.
    num_width_embeddings : ``int``, optional (default = None).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : ``int``, optional (default = None).
        The embedding size for the span_width features.
    bucket_widths : ``bool``, optional (default = False).
        Whether to bucket the span widths into log-space buckets. If ``False``,
        the raw span widths are used.
    use_exclusive_start_indices : ``bool``, optional (default = ``False``).
        If ``True``, the start indices extracted are converted to exclusive indices. Sentinels
        are used to represent exclusive span indices for the elements in the first
        position in the sequence (as the exclusive indices for these elements are outside
        of the the sequence boundary) so that start indices can be exclusive.
        NOTE: This option can be helpful to avoid the pathological case in which you
        want span differences for length 1 spans - if you use inclusive indices, you
        will end up with an ``x - x`` operation for length 1 spans, which is not good.
    """
    def __init__(self,
                 input_dim: int,
                 combination: str = 'x,y',
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            self._start_sentinel = nn.Parameter(torch.randn([1, 1, int(input_dim)]))

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError('To use a span width embedding representation, you must'
                                     'specify both num_width_buckets and span_width_embedding_dim.')
        else:
            self._span_width_embedding = None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = util.get_combined_dim(self._combination, [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        if not self._use_exclusive_start_indices:
            start_embeddings = util.batched_index_select(sequence_tensor, span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

        else:
            # We want `exclusive` span starts, so we remove 1 from the forward span starts
            # as the AllenNLP ``SpanField`` is inclusive.
            # shape (batch_size, num_spans)
            exclusive_span_starts = span_starts - 1
            # shape (batch_size, num_spans, 1)
            start_sentinel_mask = (exclusive_span_starts == -1).long().unsqueeze(-1)
            exclusive_span_starts = exclusive_span_starts * (1 - start_sentinel_mask.squeeze(-1))

            # We'll check the indices here at runtime, because it's difficult to debug
            # if this goes wrong and it's tricky to get right.
            if (exclusive_span_starts < 0).any():
                raise ValueError(f'Adjusted span indices must lie inside the the sequence tensor, '
                                 f'but found: exclusive_span_starts: {exclusive_span_starts}.')

            start_embeddings = util.batched_index_select(sequence_tensor, exclusive_span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

            # We're using sentinels, so we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with the start sentinel.
            float_start_sentinel_mask = start_sentinel_mask.float()
            start_embeddings = start_embeddings * (1 - float_start_sentinel_mask) \
                                        + float_start_sentinel_mask * self._start_sentinel

        combined_tensors = util.combine_tensors(self._combination, [start_embeddings, end_embeddings])
        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = util.bucket_values(span_ends - span_starts,
                                                 num_total_buckets=self._num_width_embeddings)
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            combined_tensors = torch.cat([combined_tensors, span_width_embeddings], -1)

        if span_indices_mask is not None:
            return combined_tensors * span_indices_mask.unsqueeze(-1).float()

        return combined_tensors


class PoolingSpanExtractor(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 combination: str = 'max',
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths
        if bucket_widths:
            raise ConfigurationError('not support')

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            raise ConfigurationError('not support')

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError('To use a span width embedding representation, you must'
                                     'specify both num_width_buckets and span_width_embedding_dim.')
        else:
            self._span_width_embedding = None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = self._input_dim
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:

        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, embedding_dim)
        # span_embeddings = util.masked_max(span_embeddings, span_mask.unsqueeze(-1), dim=2)
        span_embeddings = util.masked_mean(span_embeddings, span_mask.unsqueeze(-1), dim=2)

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(span_widths.squeeze(-1))
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        return span_embeddings


class ConvSpanExtractor(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 combination: str = 'max',
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._filter_size = int(combination.split(',')[0])
        if self._filter_size % 2 != 1:
            raise ConfigurationError('The filter size must be an odd.')
        self._combination = combination.split(',')[1]
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths
        if bucket_widths:
            raise ConfigurationError('not support')

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            raise ConfigurationError('not support')

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError('To use a span width embedding representation, you must'
                                     'specify both num_width_buckets and span_width_embedding_dim.')
        else:
            self._span_width_embedding = None

        self._conv = Conv1d(self._input_dim,
                            self._input_dim,
                            kernel_size=self._filter_size,
                            padding=int(floor(self._filter_size / 2)))
        xavier_normal_(self._conv.weight)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = self._input_dim
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:
        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        masked_span_embeddings = span_embeddings * span_mask.unsqueeze(-1)

        batch_size, num_spans, max_batch_span_width, embedding_dim = masked_span_embeddings.size()
        # Shape: (batch_size*num_spans, embedding_dim, max_batch_span_width)
        masked_span_embeddings = masked_span_embeddings.view(batch_size * num_spans, max_batch_span_width,
                                                             embedding_dim).transpose(1, 2)

        # Shape: (batch_size, embedding_dim, num_spans*max_batch_span_width)
        conv_span_embeddings = torch.nn.functional.relu(self._conv(masked_span_embeddings))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        conv_span_embeddings = conv_span_embeddings.transpose(1, 2).view(batch_size, num_spans, max_batch_span_width,
                                                                         embedding_dim)

        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = util.masked_max(conv_span_embeddings, span_mask.unsqueeze(-1), dim=2)

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(span_widths.squeeze(-1))
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        return span_embeddings


class AttentionSpanExtractor(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 combination: str = 'max',
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths
        if bucket_widths:
            raise ConfigurationError('not support')

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            raise ConfigurationError('not support')

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError('To use a span width embedding representation, you must'
                                     'specify both num_width_buckets and span_width_embedding_dim.')
        else:
            self._span_width_embedding = None

        # the allennlp SelfAttentiveSpanExtractor doesn't include span width embedding.
        self._self_attentive = SelfAttentiveSpanExtractor(self._input_dim)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = self._input_dim
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:

        span_embeddings = self._self_attentive(sequence_tensor, span_indices, sequence_mask, span_indices_mask)

        if self._span_width_embedding is not None:
            # both of shape (batch_size, num_spans, 1)
            span_starts, span_ends = span_indices.split(1, dim=-1)
            # shape (batch_size, num_spans, 1)
            # These span widths are off by 1, because the span ends are `inclusive`.
            span_widths = span_ends - span_starts
            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(span_widths.squeeze(-1))
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        return span_embeddings


class RnnSpanExtractor(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 combination: str = 'x,y',
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_exclusive_start_indices: bool = False) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._combination = combination

        self._encoder = PytorchSeq2SeqWrapper(
            StackedBidirectionalLstm(self._input_dim, int(floor(self._input_dim / 2)), 1))
        self._span_extractor = BidirectionalEndpointSpanExtractor(self._input_dim, 'y', 'y', num_width_embeddings,
                                                                  span_width_embedding_dim, bucket_widths)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._span_extractor.get_output_dim()

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> None:

        # Shape: (batch_size, sequence_length, embedding_dim)
        encoder_sequence_tensor = self._encoder(sequence_tensor, sequence_mask)

        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = self._span_extractor(encoder_sequence_tensor, span_indices, sequence_mask, span_indices_mask)

        return span_embeddings
