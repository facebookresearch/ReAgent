#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.modules.transformer as transformer
from reagent import types as rlt
from reagent.core.configuration import param_hash
from reagent.core.dataclasses import dataclass
from reagent.model_utils.seq2slate_utils import (
    DECODER_START_SYMBOL,
    PADDING_SYMBOL,
    Seq2SlateMode,
    EPSILON,
    Seq2SlateOutputArch,
    attention,
    pytorch_decoder_mask,
    clones,
    mask_logits_by_idx,
    per_symbol_to_per_seq_probs,
    print_model_info,
)
from reagent.models.base import ModelBase
from reagent.torch_utils import gather
from torch.nn.parallel.distributed import DistributedDataParallel


logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """ Candidate generation """

    def __init__(self, dim_model, candidate_size, temperature):
        super().__init__()
        self.dim_model = dim_model
        self.candidate_size = candidate_size
        self.temperature = temperature

    def forward(self, probs, greedy):
        """
        Decode one-step

        :param probs: probability distributions of decoder.
            Shape: batch_size, tgt_seq_len, candidate_size
        :param greedy: whether to greedily pick or sample the next symbol
        """
        batch_size = probs.shape[0]
        # get the last step probs shape: batch_size, candidate_size
        prob = probs[:, -1, :]
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
        super().__init__()
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class Encoder(nn.Module):
    "Core encoder is a stack of num_layers layers"

    def __init__(self, layer, num_layers):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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


class DecoderLastLayerPytorch(transformer.TransformerDecoderLayer):
    """
    The last layer of Decoder.
    Modified from PyTorch official code: instead of attention embedding,
    return attention weights which can be directly used to sample items
    """

    def forward(
        self,
        tgt,
        memory,
        tgt_mask,
        memory_mask,
    ):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        _, attn_weights = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
        )
        return attn_weights


class DecoderPyTorch(nn.Module):
    """ Transformer-based decoder based on PyTorch official implementation """

    def __init__(self, dim_model, num_heads, dim_feedforward, num_layers):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList(
            [
                transformer.TransformerDecoderLayer(
                    d_model=dim_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=0,
                )
                for _ in range(num_layers - 1)
            ]
            + [
                DecoderLastLayerPytorch(
                    d_model=dim_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=0,
                )
            ]
        )
        self.num_layers = num_layers

    def forward(self, tgt_embed, memory, tgt_src_mask, tgt_tgt_mask):
        # tgt_embed shape: batch_size, tgt_seq_len, dim_model
        # memory shape: batch_size, src_seq_len, dim_model
        # tgt_src_mask shape: batch_size, tgt_seq_len, src_seq_len
        # tgt_tgt_mask shape: batch_size, tgt_seq_len, tgt_seq_len

        # Adapt to PyTorch format
        tgt_embed = tgt_embed.transpose(0, 1)
        memory = memory.transpose(0, 1)

        output = tgt_embed

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_tgt_mask,
                memory_mask=tgt_src_mask,
            )

        batch_size, tgt_seq_len, _ = output.shape
        probs_for_placeholders = torch.zeros(
            batch_size, tgt_seq_len, 2, device=output.device
        )
        probs = torch.cat((probs_for_placeholders, output), dim=2)
        return probs


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim_model):
        """ Take in model size and number of heads """
        super().__init__()
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
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_model, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, dim_model),
        )

    def forward(self, x):
        return self.net(x)


class Embedder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        # x: raw input features. Shape: batch_size, seq_len, dim_in
        output = self.linear(x) * math.sqrt(self.dim_out)
        # output shape: batch_size, seq_len, dim_out
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, dim_model)

    def forward(self, x):
        device = x.device
        batch_size, seq_len, _ = x.shape
        position_idx = (
            torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        )
        x = x + self.pos_embed(position_idx)
        return x


class BaselineNet(nn.Module):
    def __init__(self, state_dim, dim_feedforward, num_stacked_layers):
        super().__init__()
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
    algorithm to optimize sequence-wise reward.

    One application example is to rank candidate feeds to a specific user such
    that the final list of feeds as a whole optimizes the user's engagement.

    Seq2Slate paper: https://arxiv.org/abs/1810.02019
    Transformer paper: https://arxiv.org/abs/1706.03762

    The model archtecture can also adapt to some variations.
    (1) The decoder can be autoregressive
    (2) The decoder can take encoder scores and perform iterative softmax (aka frechet sort)
    (3) No decoder and the output order is solely based on encoder scores
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
        output_arch: Seq2SlateOutputArch,
        temperature: float = 1.0,
        state_embed_dim: Optional[int] = None,
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
        :param output_arch: determines seq2slate output architecture
        :param temperature: temperature used in decoder sampling
        :param state_embed_dim: embedding dimension of state features.
            by default (if not specified), state_embed_dim = dim_model / 2
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
        self.output_arch = output_arch
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
        if self.output_arch == Seq2SlateOutputArch.FRECHET_SORT:
            # Compute score at each encoder step
            self.encoder_scorer = nn.Linear(dim_model, 1)
            # Generator needs to know the output symbol size,
            # Possible output symbols include candidate indices, decoder-start symbol
            # and padding symbol
            self.generator = Generator(dim_model, max_src_seq_len + 2, temperature)
        elif self.output_arch == Seq2SlateOutputArch.AUTOREGRESSIVE:
            self.decoder = DecoderPyTorch(
                dim_model, num_heads, dim_feedforward, num_stacked_layers
            )
            self.positional_encoding_decoder = PositionalEncoding(
                dim_model, max_len=max_tgt_seq_len
            )
            self.generator = Generator(dim_model, max_src_seq_len + 2, temperature)
        elif self.output_arch == Seq2SlateOutputArch.ENCODER_SCORE:
            # Compute score at each encoder step
            self.encoder_scorer = nn.Linear(dim_model, 1)

        if state_embed_dim is None:
            state_embed_dim = dim_model // 2
        candidate_embed_dim = dim_model - state_embed_dim
        self.state_embedder = Embedder(state_dim, state_embed_dim)
        self.candidate_embedder = Embedder(candidate_dim, candidate_embed_dim)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        print_model_info(self)

    __constants__ = [
        "state_dim",
        "candidate_dim",
        "num_stacked_layers",
        "num_heads",
        "dim_model",
        "dim_feedforward",
        "max_src_seq_len",
        "max_tgt_seq_len",
        "output_path",
        "temperature",
        "state_embed_dim",
        "_DECODER_START_SYMBOL",
        "_PADDING_SYMBOL",
        "_RANK_MODE",
        "_PER_SYMBOL_LOG_PROB_DIST_MODE",
        "_PER_SEQ_LOG_PROB_MODE",
        "_DECODE_ONE_STEP_MODE",
        "_ENCODER_SCORE_MODE",
    ]

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
            assert self.output_arch == Seq2SlateOutputArch.ENCODER_SCORE
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

        ranked_per_symbol_probs = torch.zeros(
            batch_size, tgt_seq_len, candidate_size, device=device
        )
        ranked_per_seq_probs = torch.zeros(batch_size, 1)

        if self.output_arch == Seq2SlateOutputArch.ENCODER_SCORE:
            # encoder_scores shape: batch_size, src_seq_len
            encoder_scores = self.encoder_scorer(memory).squeeze(dim=2)
            tgt_out_idx = torch.argsort(encoder_scores, dim=1, descending=True)[
                :, :tgt_seq_len
            ]
            # +2 to account for start symbol and padding symbol
            tgt_out_idx += 2
            # every position has propensity of 1 because we are just using argsort
            ranked_per_symbol_probs = ranked_per_symbol_probs.scatter(
                2, tgt_out_idx.unsqueeze(2), 1.0
            )
            ranked_per_seq_probs[:, :] = 1.0
            return ranked_per_symbol_probs, ranked_per_seq_probs, tgt_out_idx

        tgt_in_idx = (
            torch.ones(batch_size, 1, device=device)
            .fill_(self._DECODER_START_SYMBOL)
            .type(torch.long)
        )

        assert greedy is not None
        for l in range(tgt_seq_len):
            tgt_in_seq = gather(candidate_features, tgt_in_idx)
            tgt_tgt_mask, tgt_src_mask = pytorch_decoder_mask(
                memory, tgt_in_idx, self.num_heads
            )
            # shape batch_size, l + 1, candidate_size
            probs = self.decode(
                memory=memory,
                state=state,
                tgt_src_mask=tgt_src_mask,
                tgt_in_idx=tgt_in_idx,
                tgt_in_seq=tgt_in_seq,
                tgt_tgt_mask=tgt_tgt_mask,
            )
            # next candidate shape: batch_size, 1
            # prob shape: batch_size, candidate_size
            next_candidate, next_candidate_sample_prob = self.generator(probs, greedy)
            ranked_per_symbol_probs[:, l, :] = next_candidate_sample_prob
            tgt_in_idx = torch.cat([tgt_in_idx, next_candidate], dim=1)

        # remove the decoder start symbol
        # tgt_out_idx shape: batch_size, tgt_seq_len
        tgt_out_idx = tgt_in_idx[:, 1:]

        ranked_per_seq_probs = per_symbol_to_per_seq_probs(
            ranked_per_symbol_probs, tgt_out_idx
        )

        # ranked_per_symbol_probs shape: batch_size, tgt_seq_len, candidate_size
        # ranked_per_seq_probs shape: batch_size, 1
        # tgt_out_idx shape: batch_size, tgt_seq_len
        return ranked_per_symbol_probs, ranked_per_seq_probs, tgt_out_idx

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

        # tgt_tgt_mask shape: batch_size * num_heads, tgt_seq_len, tgt_seq_len
        # tgt_src_mask shape: batch_size * num_heads, tgt_seq_len, src_seq_len
        tgt_tgt_mask, tgt_src_mask = pytorch_decoder_mask(
            encoder_output, tgt_in_idx, self.num_heads
        )
        # decoder_probs shape: batch_size, tgt_seq_len, candidate_size
        decoder_probs = self.decode(
            memory=encoder_output,
            state=state,
            tgt_src_mask=tgt_src_mask,
            tgt_in_idx=tgt_in_idx,
            tgt_in_seq=tgt_in_seq,
            tgt_tgt_mask=tgt_tgt_mask,
        )
        # log_probs shape:
        # if mode == PER_SEQ_LOG_PROB_MODE: batch_size, 1
        # if mode == PER_SYMBOL_LOG_PROB_DIST_MODE: batch_size, tgt_seq_len, candidate_size
        if mode == Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE:
            per_symbol_log_probs = torch.log(torch.clamp(decoder_probs, min=EPSILON))
            return per_symbol_log_probs

        per_seq_log_probs = torch.log(
            per_symbol_to_per_seq_probs(decoder_probs, tgt_out_idx)
        )
        return per_seq_log_probs

    def encoder_output_to_scores(self, state, src_seq, src_src_mask, tgt_out_idx):
        # encoder_output shape: batch_size, src_seq_len, dim_model
        encoder_output = self.encode(state, src_seq, src_src_mask)

        # encoder_output shape: batch_size, src_seq_len, dim_model
        # tgt_out_idx shape: batch_size, tgt_seq_len
        batch_size, tgt_seq_len = tgt_out_idx.shape

        # order encoder_output by tgt_out_idx
        # slate_encoder_output shape: batch_size, tgt_seq_len, dim_model
        slate_encoder_output = gather(encoder_output, tgt_out_idx - 2)
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
        src_embed = torch.cat((state_embed, candidate_embed), dim=2)

        # encoder_output shape: batch_size, src_seq_len, dim_model
        return self.encoder(src_embed, src_mask)

    def decode(self, memory, state, tgt_src_mask, tgt_in_idx, tgt_in_seq, tgt_tgt_mask):
        # memory is the output of the encoder, the attention of each input symbol
        # memory shape: batch_size, src_seq_len, dim_model
        # tgt_in_idx shape: batch_size, tgt_seq_len
        # tgt_src_mask shape: batch_size, tgt_seq_len, src_seq_len
        # tgt_seq shape: batch_size, tgt_seq_len, dim_candidate
        # tgt_tgt_mask shape: batch_size, tgt_seq_len, tgt_seq_len
        batch_size, src_seq_len, _ = memory.shape
        _, tgt_seq_len = tgt_in_idx.shape
        candidate_size = src_seq_len + 2

        if self.output_arch == Seq2SlateOutputArch.FRECHET_SORT:
            # encoder_scores shape: batch_size, src_seq_len
            encoder_scores = self.encoder_scorer(memory).squeeze(dim=2)
            logits = torch.zeros(batch_size, tgt_seq_len, candidate_size).to(
                encoder_scores.device
            )
            logits[:, :, :2] = float("-inf")
            logits[:, :, 2:] = encoder_scores.repeat(1, tgt_seq_len).reshape(
                batch_size, tgt_seq_len, src_seq_len
            )
            logits = mask_logits_by_idx(logits, tgt_in_idx)
            probs = torch.softmax(logits, dim=2)
        elif self.output_arch == Seq2SlateOutputArch.AUTOREGRESSIVE:
            # candidate_embed shape: batch_size, tgt_seq_len, dim_model/2
            candidate_embed = self.candidate_embedder(tgt_in_seq)
            # state_embed: batch_size, dim_model/2
            state_embed = self.state_embedder(state)
            # state_embed: batch_size, tgt_seq_len, dim_model/2
            state_embed = state_embed.repeat(1, tgt_seq_len).reshape(
                batch_size, tgt_seq_len, -1
            )
            # tgt_embed: batch_size, tgt_seq_len, dim_model
            tgt_embed = self.positional_encoding_decoder(
                torch.cat((state_embed, candidate_embed), dim=2)
            )
            # output of decoder is probabilities over symbols.
            # shape: batch_size, tgt_seq_len, candidate_size
            probs = self.decoder(tgt_embed, memory, tgt_src_mask, tgt_tgt_mask)

        return probs


@dataclass
class Seq2SlateNet(ModelBase):
    __hash__ = param_hash

    state_dim: int
    candidate_dim: int
    num_stacked_layers: int
    dim_model: int
    max_src_seq_len: int
    max_tgt_seq_len: int
    output_arch: Seq2SlateOutputArch
    temperature: float

    def __post_init_post_parse__(self) -> None:
        super().__init__()
        # pyre-fixme[16]: `Seq2SlateNet` has no attribute `seq2slate`.
        self.seq2slate = self._build_model()

    def _build_model(self):
        return None

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
        # pyre-fixme[16]: `Seq2SlateNet` has no attribute `seq2slate`.
        res = self.seq2slate(input, mode=mode, tgt_seq_len=tgt_seq_len, greedy=greedy)
        if mode == Seq2SlateMode.RANK_MODE:
            return rlt.RankingOutput(
                ranked_per_symbol_probs=res[0],
                ranked_per_seq_probs=res[1],
                ranked_tgt_out_idx=res[2],
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

    def get_distributed_data_parallel_model(self):
        return _DistributedSeq2SlateNet(self)


@dataclass
class Seq2SlateTransformerNet(Seq2SlateNet):
    __hash__ = param_hash

    num_heads: int
    dim_feedforward: int
    state_embed_dim: Optional[int] = None

    def _build_model(self):
        return Seq2SlateTransformerModel(
            state_dim=self.state_dim,
            candidate_dim=self.candidate_dim,
            num_stacked_layers=self.num_stacked_layers,
            num_heads=self.num_heads,
            dim_model=self.dim_model,
            dim_feedforward=self.dim_feedforward,
            max_src_seq_len=self.max_src_seq_len,
            max_tgt_seq_len=self.max_tgt_seq_len,
            output_arch=self.output_arch,
            temperature=self.temperature,
            state_embed_dim=self.state_embed_dim,
        )


class _DistributedSeq2SlateNet(ModelBase):
    def __init__(self, seq2slate_net: Seq2SlateNet):
        super().__init__()

        current_device = torch.cuda.current_device()
        self.data_parallel = DistributedDataParallel(
            # pyre-fixme[16]: `Seq2SlateNet` has no attribute `seq2slate`.
            seq2slate_net.seq2slate,
            device_ids=[current_device],
            output_device=current_device,
        )
        self.seq2slate_net = seq2slate_net

    def input_prototype(self):
        return self.seq2slate_net.input_prototype()

    def cpu_model(self):
        return self.seq2slate_net.cpu_model()

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
                ranked_per_symbol_probs=res[0],
                ranked_per_seq_probs=res[1],
                ranked_tgt_out_idx=res[2],
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
