#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging
import math
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent import types as rlt
from reagent.models.base import ModelBase
from torch.nn.parallel.distributed import DistributedDataParallel


logger = logging.getLogger(__name__)


class Seq2SlateMode(Enum):
    RANK_MODE = "rank"
    PER_SEQ_LOG_PROB_MODE = "per_sequence_log_prob"
    PER_SYMBOL_LOG_PROB_DIST_MODE = "per_symbol_log_prob_dist"
    DECODE_ONE_STEP_MODE = "decode_one_step"
    ENCODER_SCORE_MODE = "encoder_score_mode"


PADDING_SYMBOL = 0
DECODER_START_SYMBOL = 1


def subsequent_mask(size, device):
    """
    Mask out subsequent positions. Mainly used in the decoding process,
    in which an item should not attend subsequent items.
    """
    attn_shape = (1, size, size)
    subsequent_mask = (
        1 - torch.triu(torch.ones(*attn_shape, device=device), diagonal=1)
    ).type(torch.int8)
    return subsequent_mask


def subsequent_and_padding_mask(tgt_in_idx):
    """ Create a mask to hide padding and future items """
    # tgt_in_idx shape: batch_size, seq_len

    # tgt_tgt_mask shape: batch_size, 1, seq_len
    tgt_tgt_mask = (tgt_in_idx != PADDING_SYMBOL).unsqueeze(-2).type(torch.int8)
    # subseq_mask shape: 1, seq_len, seq_len
    subseq_mask = subsequent_mask(tgt_in_idx.size(-1), tgt_in_idx.device)
    # tgt_tgt_mask shape: batch_size, seq_len, seq_len
    tgt_tgt_mask = tgt_tgt_mask & subseq_mask
    return tgt_tgt_mask


def clones(module, N):
    """
    Produce N identical layers.

    :param module: nn.Module class
    :param N: number of copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask, d_k):
    """ Scaled Dot Product Attention """
    # mask shape: batch_size x 1 x seq_len x seq_len

    # scores shape: batch_size x num_heads x seq_len x seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn shape: batch_size x num_heads x seq_len x seq_len
    p_attn = F.softmax(scores, dim=3)
    # attn shape: batch_size x num_heads x seq_len x d_k
    attn = torch.matmul(p_attn, value)
    return attn, p_attn


class Generator(nn.Module):
    """ Define standard linear + softmax generation step. """

    def __init__(self, dim_model, candidate_size):
        super(Generator, self).__init__()
        self.dim_model = dim_model
        self.proj = nn.Linear(dim_model, candidate_size)

    def forward(self, mode, decoder_output=None, tgt_in_idx=None, greedy=None):
        if mode in (
            Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE,
            Seq2SlateMode.PER_SEQ_LOG_PROB_MODE,
        ):
            return self._log_probs(decoder_output, tgt_in_idx, mode)
        elif mode == Seq2SlateMode.DECODE_ONE_STEP_MODE:
            assert greedy is not None
            return self._decode_one_step(decoder_output, tgt_in_idx, greedy)
        else:
            raise NotImplementedError()

    def _log_probs(self, x, tgt_in_idx, mode):
        """
        Return the log probability distribution at each decoding step

        :param x: the output of decoder. Shape: batch_size, seq_len, dim_model
        :param tgt_idx: the indices of candidates in decoder input sequences.
            The first symbol is always DECODER_START_SYMBOL.
            Shape: batch_size, seq_len
        """
        assert mode in (
            Seq2SlateMode.PER_SEQ_LOG_PROB_MODE,
            Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE,
        )
        # logits: the probability distribution of each symbol
        # batch_size, seq_len, candidate_size
        logits = self.proj(x)
        # the first two symbols are reserved for padding and decoder-starting symbols
        # so they should never be a possible output label
        logits[:, :, :2] = float("-inf")

        if mode == Seq2SlateMode.PER_SEQ_LOG_PROB_MODE:
            batch_size, seq_len = tgt_in_idx.shape
            mask_indices = torch.tril(
                tgt_in_idx.repeat(1, seq_len).reshape(batch_size, seq_len, seq_len),
                diagonal=0,
            )
            logits.scatter_(2, mask_indices, float("-inf"))

        # log_probs shape: batch_size, seq_len, candidate_size
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def _decode_one_step(self, x, tgt_in_idx, greedy):
        """
        Decode one-step

        :param x: the output of the decoder. Shape: batch_size, seq_len, dim_model
        :param tgt_in_idx: input to the decoder, the first symbol is always the
            starting symbol. Shape: batch_size, seq_len
        :param greedy: whether to greedily pick or sample the next symbol
        """
        # get the last step of decoder output
        last_step_x = x[:, -1, :]

        batch_size = x.shape[0]
        # logits shape: batch_size, candidate_size
        logits = self.proj(last_step_x)
        # invalidate the padding symbol and decoder-starting symbol
        logits[:, :2] = float("-inf")
        # invalidate symbols already appeared in decoded sequences
        logits.scatter_(1, tgt_in_idx, float("-inf"))
        prob = F.softmax(logits, dim=1)
        if greedy:
            _, next_candidate = torch.max(prob, dim=1)
        else:
            next_candidate = torch.multinomial(prob, num_samples=1, replacement=False)
        next_candidate = next_candidate.reshape(batch_size, 1)

        # next_candidate: the decoded symbols for the latest step
        # shape: batch_size x 1
        # prob: generative probabilities of the latest step
        # shape: batch_size x candidate_size
        return next_candidate, prob


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, dim_model):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class Encoder(nn.Module):
    "Core encoder is a stack of num_layers layers"

    def __init__(self, layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.dim_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """ Encoder is made up of self-attn and feed forward """

    def __init__(self, dim_model, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(dim_model), 2)
        self.dim_model = dim_model

    def forward(self, src_embed, src_mask):
        # src_embed shape: batch_size, seq_len, dim_model
        # src_src_mask shape: batch_size, seq_len, seq_len

        def self_attn_layer(x):
            return self.self_attn(x, x, x, src_mask)

        # attn_output shape: batch_size, seq_len, dim_model
        attn_output = self.sublayer[0](src_embed, self_attn_layer)
        # return shape: batch_size, seq_len, dim_model
        return self.sublayer[1](attn_output, self.feed_forward)


class Decoder(nn.Module):
    """ Generic num_layers layer decoder with masking."""

    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, tgt_src_mask, tgt_tgt_mask):
        # each layer is one DecoderLayer
        for layer in self.layers:
            x = layer(x, memory, tgt_src_mask, tgt_tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """ Decoder is made of self-attn, src-attn, and feed forward """

    def __init__(self, size, self_attn, src_attn, feed_forward):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 3)

    def forward(self, x, m, tgt_src_mask, tgt_tgt_mask):
        # x is target embedding or the output of previous decoder layer
        # x shape: batch_size, seq_len, dim_model
        # m is the output of the last encoder layer
        # m shape: batch_size, seq_len, dim_model
        # tgt_src_mask shape: batch_size, tgt_seq_len, src_seq_len
        # tgt_tgt_mask shape: batch_size, tgt_seq_len, tgt_seq_len
        def self_attn_layer_tgt(x):
            return self.self_attn(query=x, key=x, value=x, mask=tgt_tgt_mask)

        def self_attn_layer_src(x):
            return self.src_attn(query=x, key=m, value=m, mask=tgt_src_mask)

        x = self.sublayer[0](x, self_attn_layer_tgt)
        x = self.sublayer[1](x, self_attn_layer_src)
        # return shape: batch_size, seq_len, dim_model
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim_model):
        """ Take in model size and number of heads """
        super(MultiHeadedAttention, self).__init__()
        assert dim_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = dim_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all num_heads heads.
            # mask shape: batch_size, 1, seq_len, seq_len
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from dim_model => num_heads x d_k
        # self.linear[0, 1, 2] is query weight matrix, key weight matrix, and
        # value weight matrix, respectively.
        # l(x) represents the transformed query matrix, key matrix and value matrix
        # l(x) has shape (batch_size, seq_len, dim_model). You can think l(x) as
        # the matrices from a one-head attention; or you can think
        # l(x).view(...).transpose(...) as the matrices of num_heads attentions,
        # each attention has d_k dimension.
        query, key, value = [
            l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: batch_size, num_heads, seq_len, d_k
        x, _ = attention(query, key, value, mask, self.d_k)

        # 3) "Concat" using a view and apply a final linear.
        # each attention's output is d_k dimension. Concat num_heads attention's outputs
        # x shape: batch_size, seq_len, dim_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_feedforward):
        super(PositionwiseFeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_model, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, dim_model),
        )

    def forward(self, x):
        return self.net(x)


class Embedder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Embedder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        # x: raw input features. Shape: batch_size, seq_len, dim_in
        output = self.linear(x) * math.sqrt(self.dim_out)
        # output shape: batch_size, seq_len, dim_out
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, dim_model, 2) * -(math.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe shape: 1, max_len, dim_model
        self.register_buffer("pe", pe)

    def forward(self, x, seq_len):
        x = x + self.pe[:, :seq_len]
        return x


class BaselineNet(nn.Module):
    def __init__(self, state_dim, dim_feedforward, num_stacked_layers):
        super(BaselineNet, self).__init__()
        nn_blocks = [nn.Linear(state_dim, dim_feedforward), nn.ReLU()]
        assert num_stacked_layers >= 1
        for _ in range(num_stacked_layers - 1):
            nn_blocks.extend([nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU()])
        nn_blocks.append(nn.Linear(dim_feedforward, 1))
        self.mlp = nn.Sequential(*nn_blocks)

    def forward(self, input: rlt.PreprocessedRankingInput):
        x = input.state.float_features
        return self.mlp(x)


class Seq2SlateTransformerModel(nn.Module):
    """
    A Seq2Slate network with Transformer. The network is essentially an
    encoder-decoder structure. The encoder inputs a sequence of candidate feature
    vectors and a state feature vector, and the decoder outputs an ordered
    list of candidate indices. The output order is learned through REINFORCE
    algorithm to optimize some sequence-wise reward which is also specific to
    the provided state feature.

    One application example is to rank candidate feeds to a specific user such
    that the final list of feeds as a whole optimizes the user's engagement.

    Seq2Slate paper: https://arxiv.org/abs/1810.02019
    Transformer paper: https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        num_stacked_layers: int,
        num_heads: int,
        dim_model: int,
        dim_feedforward: int,
        max_src_seq_len: int,
        max_tgt_seq_len: int,
        encoder_only: bool,
    ):
        """
        :param state_dim: state feature dimension
        :param candidate_dim: candidate feature dimension
        :param num_stacked_layers: number of stacked layers in Transformer
        :param num_heads: number of attention heads used in Transformer
        :param dim_model: number of attention dimensions in Transformer
        :param dim_feedforward: number of hidden units in FeedForward layers
            in Transformer
        :param max_src_seq_len: the maximum length of input sequences
        :param max_tgt_seq_len: the maximum length of output sequences
        :param encoder_only: if True, the model only has an Encoder but no Decoder.
        """
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.num_stacked_layers = num_stacked_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len
        self.encoder_only = encoder_only
        self._DECODER_START_SYMBOL = DECODER_START_SYMBOL
        self._PADDING_SYMBOL = PADDING_SYMBOL
        self._RANK_MODE = Seq2SlateMode.RANK_MODE
        self._PER_SYMBOL_LOG_PROB_DIST_MODE = (
            Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE
        )
        self._PER_SEQ_LOG_PROB_MODE = Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
        self._DECODE_ONE_STEP_MODE = Seq2SlateMode.DECODE_ONE_STEP_MODE
        self._ENCODER_SCORE_MODE = Seq2SlateMode.ENCODER_SCORE_MODE

        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, dim_model)
        ff = PositionwiseFeedForward(dim_model, dim_feedforward)
        self.encoder = Encoder(
            EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers
        )
        if self.encoder_only:
            # score encoder output
            self.encoder_scorer = nn.Linear(dim_model, 1)
        else:
            self.decoder = Decoder(
                DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers
            )
            # Generator needs to know the output symbol size,
            # Possible output symbols include candidate indices, decoder-start symbol
            # and padding symbol
            self.generator = Generator(dim_model, max_src_seq_len + 2)
        self.candidate_embedder = Embedder(candidate_dim, dim_model // 2)
        self.state_embedder = Embedder(state_dim, dim_model // 2)
        self.positional_encoding = PositionalEncoding(
            dim_model, max_len=2 * (max_src_seq_len + max_tgt_seq_len)
        )
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self._print_model_info()

    __constants__ = [
        "state_dim",
        "candidate_dim",
        "num_stacked_layers",
        "num_heads",
        "dim_model",
        "dim_feedforward",
        "max_src_seq_len",
        "max_tgt_seq_len",
        "encoder_only",
        "_DECODER_START_SYMBOL",
        "_PADDING_SYMBOL",
        "_RANK_MODE",
        "_PER_SYMBOL_LOG_PROB_DIST_MODE",
        "_PER_SEQ_LOG_PROB_MODE",
        "_DECODE_ONE_STEP_MODE",
        "_ENCODER_SCORE_MODE",
    ]

    def _print_model_info(self):
        def _num_of_params(model):
            return len(torch.cat([p.flatten() for p in model.parameters()]))

        logger.info(f"Num of total params: {_num_of_params(self)}")
        logger.info(f"Num of Encoder params: {_num_of_params(self.encoder)}")
        logger.info(
            f"Num of Candidate Embedder params: {_num_of_params(self.candidate_embedder)}"
        )
        logger.info(
            f"Num of State Embedder params: {_num_of_params(self.state_embedder)}"
        )
        if self.encoder_only:
            logger.info(
                f"Num of Encoder_Scorer params: {_num_of_params(self.encoder_scorer)}"
            )
        else:
            logger.info(f"Num of Decoder params: {_num_of_params(self.decoder)}")
            logger.info(f"Num of Generator params: {_num_of_params(self.generator)}")

    def forward(
        self,
        input: rlt.PreprocessedRankingInput,
        mode: str,
        tgt_seq_len: Optional[int] = None,
        greedy: Optional[bool] = None,
    ):
        """
        :param input: model input
        :param mode: a string indicating which mode to perform.
            "rank": return ranked actions and their generative probabilities.
            "per_seq_log_probs": return generative log probabilities of given
                tgt sequences (used for REINFORCE training)
            "per_symbol_log_probs": return generative log probabilties of each
                symbol in given tgt sequences (used in TEACHER FORCING and
                DIFFERENTIABLE_REWARD training)
        :param tgt_seq_len: the length of output sequence to be decoded. Only used
            in rank mode
        :param greedy: whether to sample based on softmax distribution or greedily
            when decoding. Only used in rank mode
        """
        if mode == self._RANK_MODE:
            if tgt_seq_len is None:
                tgt_seq_len = self.max_tgt_seq_len
            return self._rank(
                state=input.state.float_features,
                src_seq=input.src_seq.float_features,
                src_src_mask=input.src_src_mask,
                tgt_seq_len=tgt_seq_len,
                greedy=greedy,
            )
        elif mode in (self._PER_SEQ_LOG_PROB_MODE, self._PER_SYMBOL_LOG_PROB_DIST_MODE):
            assert input.tgt_in_seq is not None
            return self._log_probs(
                state=input.state.float_features,
                src_seq=input.src_seq.float_features,
                # pyre-fixme[16]: `Optional` has no attribute `float_features`.
                tgt_in_seq=input.tgt_in_seq.float_features,
                src_src_mask=input.src_src_mask,
                tgt_tgt_mask=input.tgt_tgt_mask,
                tgt_in_idx=input.tgt_in_idx,
                tgt_out_idx=input.tgt_out_idx,
                mode=mode,
            )
        elif mode == self._ENCODER_SCORE_MODE:
            assert self.encoder_only
            return self.encoder_output_to_scores(
                state=input.state.float_features,
                src_seq=input.src_seq.float_features,
                src_src_mask=input.src_src_mask,
                tgt_out_idx=input.tgt_out_idx,
            )

    def _rank(self, state, src_seq, src_src_mask, tgt_seq_len, greedy):
        """ Decode sequences based on given inputs """
        device = src_seq.device
        batch_size, src_seq_len, candidate_dim = src_seq.shape
        candidate_size = src_seq_len + 2

        # candidate_features is used as look-up table for candidate features.
        # the second dim is src_seq_len + 2 because we also want to include
        # features of start symbol and padding symbol
        candidate_features = torch.zeros(
            batch_size, src_seq_len + 2, candidate_dim, device=device
        )
        # TODO: T62502977 create learnable feature vectors for start symbol
        # and padding symbol
        candidate_features[:, 2:, :] = src_seq

        # memory shape: batch_size, src_seq_len, dim_model
        memory = self.encode(state, src_seq, src_src_mask)

        if self.encoder_only:
            # encoder_scores shape: batch_size, tgt_seq_len
            encoder_scores = self.encoder_scorer(memory).squeeze(dim=2)
            tgt_out_idx = torch.argsort(encoder_scores, dim=1, descending=True)[
                :, :tgt_seq_len
            ]
            # +2 to account for start symbol and padding symbol
            tgt_out_idx += 2
            # every position has propensity of 1 because we are just using argsort
            tgt_out_probs = torch.ones(
                batch_size, tgt_seq_len, candidate_size, device=device
            )

            # TODO: T62503033 return encoder_scores so that we can apply
            # frechet policy gradient

            return tgt_out_probs, tgt_out_idx

        tgt_in_idx = (
            torch.ones(batch_size, 1, device=device)
            .fill_(self._DECODER_START_SYMBOL)
            .type(torch.long)
        )
        tgt_out_probs = torch.zeros(
            batch_size, tgt_seq_len, candidate_size, device=device
        )
        assert greedy is not None
        for l in range(tgt_seq_len):
            tgt_in_seq = (
                candidate_features[
                    torch.arange(batch_size, device=device).repeat_interleave(l + 1),
                    tgt_in_idx.flatten(),
                ]
                .view(batch_size, l + 1, -1)
                .to(device)
            )
            tgt_src_mask = src_src_mask[:, : l + 1, :]
            out = self.decode(
                memory=memory,
                state=state,
                tgt_src_mask=tgt_src_mask,
                tgt_in_seq=tgt_in_seq,
                tgt_tgt_mask=subsequent_mask(l + 1, device),
                tgt_seq_len=l + 1,
            )
            # next candidate shape: batch_size, 1
            # prob shape: batch_size, candidate_size
            next_candidate, prob = self.generator(
                mode=self._DECODE_ONE_STEP_MODE,
                decoder_output=out,
                tgt_in_idx=tgt_in_idx,
                greedy=greedy,
            )
            tgt_out_probs[:, l, :] = prob
            tgt_in_idx = torch.cat([tgt_in_idx, next_candidate], dim=1)

        # remove the decoder start symbol
        # tgt_out_idx shape: batch_size, tgt_seq_len
        tgt_out_idx = tgt_in_idx[:, 1:]
        # tgt_out_probs shape: batch_size, tgt_seq_len, candidate_size
        return tgt_out_probs, tgt_out_idx

    def _log_probs(
        self,
        state,
        src_seq,
        tgt_in_seq,
        src_src_mask,
        tgt_tgt_mask,
        tgt_in_idx,
        tgt_out_idx,
        mode,
    ):
        """
        Compute log of generative probabilities of given tgt sequences
        (used for REINFORCE training)
        """
        # encoder_output shape: batch_size, src_seq_len, dim_model
        encoder_output = self.encode(state, src_seq, src_src_mask)

        tgt_seq_len = tgt_in_seq.shape[1]
        src_seq_len = src_seq.shape[1]
        assert tgt_seq_len <= src_seq_len

        # tgt_src_mask shape: batch_size, tgt_seq_len, src_seq_len
        tgt_src_mask = src_src_mask[:, :tgt_seq_len, :]

        # decoder_output shape: batch_size, tgt_seq_len, dim_model
        decoder_output = self.decode(
            memory=encoder_output,
            state=state,
            tgt_src_mask=tgt_src_mask,
            tgt_in_seq=tgt_in_seq,
            tgt_tgt_mask=tgt_tgt_mask,
            tgt_seq_len=tgt_seq_len,
        )
        # log_probs shape:
        # if mode == PER_SEQ_LOG_PROB_MODE: batch_size
        # if mode == PER_SYMBOL_LOG_PROB_DIST_MODE: batch_size, tgt_seq_len, candidate_size
        log_probs = self._decoder_output_to_log_probs(
            decoder_output, tgt_in_idx, tgt_out_idx, mode
        )

        return log_probs

    def _decoder_output_to_log_probs(
        self, decoder_output, tgt_in_idx, tgt_out_idx, mode
    ):
        """
        :param decoder_output: the output from the decoder, with shape:
            (batch_size, seq_len, dim_model)
        :param tgt_in_idx: input idx to the decoder, the first symbol is
            always the DECODER_START_SYMBOL. Shape: batch_size x seq_len
        :param tgt_out_idx: output idx of the decoder. Shape: batch_size x seq_len
        :param mode: return log prob distribution per symbol or reduce them per sequence
        """
        assert mode in (
            self._PER_SEQ_LOG_PROB_MODE,
            self._PER_SYMBOL_LOG_PROB_DIST_MODE,
        )
        # per_symbol_log_probs: log probability distribution of each symbol
        # shape: batch_size, seq_len, candidate_size
        per_symbol_log_probs = self.generator(
            mode=mode, decoder_output=decoder_output, tgt_in_idx=tgt_in_idx
        )
        if mode == self._PER_SYMBOL_LOG_PROB_DIST_MODE:
            return per_symbol_log_probs

        # shape: batch_size, 1
        return self.per_symbol_to_per_seq_log_probs(per_symbol_log_probs, tgt_out_idx)

    @staticmethod
    def per_symbol_to_per_seq_log_probs(per_symbol_log_probs, tgt_out_idx):
        device = per_symbol_log_probs.device
        batch_size, seq_len, candidate_size = per_symbol_log_probs.shape
        # log_probs: log probability of each symbol in the tgt_out_idx
        # shape: batch_size, seq_len
        log_probs = per_symbol_log_probs.view(-1, candidate_size)[
            torch.arange(batch_size * seq_len, device=device), tgt_out_idx.flatten()
        ].view(batch_size, seq_len)
        # shape: batch_size, 1
        return log_probs.sum(dim=1, keepdim=True)

    def encoder_output_to_scores(self, state, src_seq, src_src_mask, tgt_out_idx):
        # encoder_output shape: batch_size, src_seq_len, dim_model
        encoder_output = self.encode(state, src_seq, src_src_mask)

        # encoder_output shape: batch_size, src_seq_len, dim_model
        # tgt_out_idx shape: batch_size, tgt_seq_len
        device = encoder_output.device
        batch_size, tgt_seq_len = tgt_out_idx.shape

        # order encoder_output by tgt_out_idx
        # slate_encoder_output shape: batch_size, tgt_seq_len, dim_model
        slate_encoder_output = encoder_output[
            torch.arange(batch_size, device=device).repeat_interleave(tgt_seq_len),
            (tgt_out_idx - 2).flatten(),
        ].reshape(batch_size, tgt_seq_len, -1)
        # encoder_scores shape: batch_size, tgt_seq_len
        return self.encoder_scorer(slate_encoder_output).squeeze()

    def encode(self, state, src_seq, src_mask):
        # state: batch_size, state_dim
        # src_seq: batch_size, src_seq_len, dim_candidate
        # src_src_mask shape: batch_size, src_seq_len, src_seq_len
        batch_size = src_seq.shape[0]

        # candidate_embed: batch_size, src_seq_len, dim_model/2
        candidate_embed = self.candidate_embedder(src_seq)
        # state_embed: batch_size, dim_model/2
        state_embed = self.state_embedder(state)
        # transform state_embed into shape: batch_size, src_seq_len, dim_model/2
        state_embed = state_embed.repeat(1, self.max_src_seq_len).reshape(
            batch_size, self.max_src_seq_len, -1
        )

        # Input at each encoder step is actually concatenation of state_embed
        # and candidate embed. state_embed is replicated at each encoding step.
        # src_embed shape: batch_size, src_seq_len, dim_model
        src_embed = self.positional_encoding(
            torch.cat((state_embed, candidate_embed), dim=2), self.max_src_seq_len
        )

        # encoder_output shape: batch_size, src_seq_len, dim_model
        return self.encoder(src_embed, src_mask)

    def decode(
        self, memory, state, tgt_src_mask, tgt_in_seq, tgt_tgt_mask, tgt_seq_len
    ):
        # memory is the output of the encoder, the attention of each input symbol
        # memory shape: batch_size, src_seq_len, dim_model
        # tgt_src_mask shape: batch_size, tgt_seq_len, src_seq_len
        # tgt_seq shape: batch_size, tgt_seq_len, dim_candidate
        # tgt_tgt_mask shape: batch_size, tgt_seq_len, tgt_seq_len
        batch_size = tgt_in_seq.shape[0]

        # candidate_embed shape: batch_size, tgt_seq_len, dim_model/2
        candidate_embed = self.candidate_embedder(tgt_in_seq)
        # state_embed: batch_size, dim_model/2
        state_embed = self.state_embedder(state)
        # state_embed: batch_size, tgt_seq_len, dim_model/2
        state_embed = state_embed.repeat(1, tgt_seq_len).reshape(
            batch_size, tgt_seq_len, -1
        )

        # tgt_embed: batch_size, tgt_seq_len, dim_model
        tgt_embed = self.positional_encoding(
            torch.cat((state_embed, candidate_embed), dim=2), tgt_seq_len
        )

        # output of decoder will be later transformed into probabilities over symbols.
        # shape: batch_size, tgt_seq_len, dim_model
        return self.decoder(tgt_embed, memory, tgt_src_mask, tgt_tgt_mask)


class Seq2SlateTransformerNet(ModelBase):
    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        num_stacked_layers: int,
        num_heads: int,
        dim_model: int,
        dim_feedforward: int,
        max_src_seq_len: int,
        max_tgt_seq_len: int,
        encoder_only: bool,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.num_stacked_layers = num_stacked_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len
        self.encoder_only = encoder_only

        self.seq2slate_transformer = Seq2SlateTransformerModel(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_stacked_layers=num_stacked_layers,
            num_heads=num_heads,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            max_src_seq_len=max_src_seq_len,
            max_tgt_seq_len=max_tgt_seq_len,
            encoder_only=encoder_only,
        )

    def get_distributed_data_parallel_model(self):
        return _DistributedSeq2SlateTransformerNet(self)

    def input_prototype(self):
        return rlt.PreprocessedRankingInput.from_tensors(
            state=torch.randn(1, self.state_dim),
            src_seq=torch.randn(1, self.max_src_seq_len, self.candidate_dim),
            tgt_in_seq=torch.randn(1, self.max_tgt_seq_len, self.candidate_dim),
            tgt_out_seq=torch.randn(1, self.max_tgt_seq_len, self.candidate_dim),
            src_src_mask=torch.ones(1, self.max_src_seq_len, self.max_src_seq_len),
            tgt_tgt_mask=torch.ones(1, self.max_tgt_seq_len, self.max_tgt_seq_len),
            slate_reward=torch.randn(1),
        )

    def forward(
        self,
        input: rlt.PreprocessedRankingInput,
        mode: str,
        tgt_seq_len: Optional[int] = None,
        greedy: Optional[bool] = None,
    ):
        res = self.seq2slate_transformer(
            input, mode=mode, tgt_seq_len=tgt_seq_len, greedy=greedy
        )
        if mode == Seq2SlateMode.RANK_MODE:
            return rlt.RankingOutput(
                ranked_tgt_out_idx=res[1], ranked_tgt_out_probs=res[0]
            )
        elif mode in (
            Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE,
            Seq2SlateMode.PER_SEQ_LOG_PROB_MODE,
        ):
            return rlt.RankingOutput(log_probs=res)
        elif mode == Seq2SlateMode.ENCODER_SCORE_MODE:
            return rlt.RankingOutput(encoder_scores=res)
        else:
            raise NotImplementedError()


class _DistributedSeq2SlateTransformerNet(ModelBase):
    def __init__(self, seq2slate_transformer_net: Seq2SlateTransformerNet):
        super().__init__()
        self.state_dim = seq2slate_transformer_net.state_dim
        self.candidate_dim = seq2slate_transformer_net.candidate_dim
        self.num_stacked_layers = seq2slate_transformer_net.num_stacked_layers
        self.num_heads = seq2slate_transformer_net.num_heads
        self.dim_model = seq2slate_transformer_net.dim_model
        self.dim_feedforward = seq2slate_transformer_net.dim_feedforward
        self.max_src_seq_len = seq2slate_transformer_net.max_src_seq_len
        self.max_tgt_seq_len = seq2slate_transformer_net.max_tgt_seq_len
        self.encoder_only = seq2slate_transformer_net.encoder_only

        current_device = torch.cuda.current_device()
        self.data_parallel = DistributedDataParallel(
            seq2slate_transformer_net.seq2slate_transformer,
            device_ids=[current_device],
            output_device=current_device,
        )
        self.seq2slate_transformer_net = seq2slate_transformer_net

    def input_prototype(self):
        return self.seq2slate_transformer_net.input_prototype()

    def cpu_model(self):
        return self.seq2slate_transformer_net.cpu_model()

    def forward(
        self,
        input: rlt.PreprocessedRankingInput,
        mode: str,
        tgt_seq_len: Optional[int] = None,
        greedy: Optional[bool] = None,
    ):
        res = self.data_parallel(
            input, mode=mode, tgt_seq_len=tgt_seq_len, greedy=greedy
        )
        if mode == Seq2SlateMode.RANK_MODE:
            return rlt.RankingOutput(
                ranked_tgt_out_idx=res[1], ranked_tgt_out_probs=res[0]
            )
        elif mode in (
            Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE,
            Seq2SlateMode.PER_SEQ_LOG_PROB_MODE,
        ):
            return rlt.RankingOutput(log_probs=res)
        elif mode == Seq2SlateMode.ENCODER_SCORE_MODE:
            return rlt.RankingOutput(encoder_scores=res)
        else:
            raise NotImplementedError()
