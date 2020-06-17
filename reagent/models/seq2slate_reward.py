#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy

import torch
import torch.nn as nn
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.seq2slate import (
    DECODER_START_SYMBOL,
    Decoder,
    DecoderLayer,
    Embedder,
    Encoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    subsequent_and_padding_mask,
)


class Seq2SlateRewardNet(ModelBase):
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
    ):
        """
        A reward network that predicts slate reward.

        It uses a transformer-based encoder to encode the items shown in the slate.
        The slate reward is predicted by attending all encoder steps' outputs.

        For convenience, Seq2SlateRewardModel and Seq2SlateTransformerModel share
        the same parameter notations. Therefore, the reward model's encoder is
        actually applied on target sequences (i.e., slates) referred in
        Seq2SlateTransformerModel.

        Note that max_src_seq_len is the
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

        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, dim_model)
        ff = PositionwiseFeedForward(dim_model, dim_feedforward)
        self.encoder = Encoder(
            EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers
        )
        self.decoder = Decoder(
            DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers
        )
        self.candidate_embedder = Embedder(candidate_dim, dim_model // 2)
        self.state_embedder = Embedder(state_dim, dim_model // 2)
        self.positional_encoding = PositionalEncoding(
            dim_model, max_len=2 * (max_src_seq_len + max_tgt_seq_len)
        )
        self.proj = nn.Linear(dim_model, 1)
        self.decoder_start_vec = nn.Parameter(
            torch.zeros(candidate_dim), requires_grad=True
        )

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

        # candidate_embed shape: batch_size, seq_len, dim_model/2
        candidate_embed = self.candidate_embedder(tgt_in_seq)
        # state_embed: batch_size, dim_model/2
        state_embed = self.state_embedder(state)
        # state_embed: batch_size, seq_len, dim_model/2
        state_embed = state_embed.repeat(1, tgt_seq_len).reshape(
            batch_size, tgt_seq_len, -1
        )

        # tgt_embed: batch_size, seq_len, dim_model
        tgt_embed = self.positional_encoding(
            torch.cat((state_embed, candidate_embed), dim=2), tgt_seq_len
        )

        # output of decoder will be later transformed into probabilities over symbols.
        # shape: batch_size, seq_len, dim_model
        return self.decoder(tgt_embed, memory, tgt_src_mask, tgt_tgt_mask)

    def input_prototype(self):
        return rlt.PreprocessedRankingInput.from_tensors(
            state=torch.randn(1, self.state_dim),
            src_seq=torch.randn(1, self.max_src_seq_len, self.candidate_dim),
            tgt_in_seq=torch.randn(1, self.max_tgt_seq_len, self.candidate_dim),
            tgt_out_seq=torch.randn(1, self.max_tgt_seq_len, self.candidate_dim),
            src_src_mask=torch.ones(1, self.max_src_seq_len, self.max_src_seq_len),
            tgt_tgt_mask=torch.ones(1, self.max_tgt_seq_len, self.max_tgt_seq_len),
            tgt_out_idx=torch.arange(self.max_tgt_seq_len).reshape(1, -1) + 2,
        )

    def _convert_seq2slate_to_reward_model_format(
        self, input: rlt.PreprocessedRankingInput
    ):
        """
        In the reward model, the transformer decoder should see the full
        sequences; while in seq2slate, the decoder only sees the sequence
        before the last item.
        """
        device = next(self.parameters()).device
        # pyre-fixme[16]: Optional type has no attribute `float_features`.
        batch_size, tgt_seq_len, candidate_dim = input.tgt_out_seq.float_features.shape
        assert self.max_tgt_seq_len == tgt_seq_len

        # shape: batch_szie, tgt_seq_len + 1
        tgt_in_idx = torch.cat(
            (
                torch.full((batch_size, 1), DECODER_START_SYMBOL, device=device).long(),
                input.tgt_out_idx,
            ),
            dim=1,
        )
        tgt_tgt_mask = subsequent_and_padding_mask(tgt_in_idx)
        # shape: batch_size, tgt_seq_len + 1, candidate_dim
        tgt_in_seq = torch.cat(
            (
                self.decoder_start_vec.repeat(batch_size, 1, 1),
                input.tgt_out_seq.float_features,
            ),
            dim=1,
        )

        return rlt.PreprocessedRankingInput.from_tensors(
            state=input.state.float_features,
            src_seq=input.src_seq.float_features,
            src_src_mask=input.src_src_mask,
            tgt_in_seq=tgt_in_seq,
            tgt_tgt_mask=tgt_tgt_mask,
        )

    def forward(self, input: rlt.PreprocessedRankingInput):
        input = self._convert_seq2slate_to_reward_model_format(input)
        state, src_seq, tgt_in_seq, src_src_mask, tgt_tgt_mask = (
            input.state.float_features,
            input.src_seq.float_features,
            input.tgt_in_seq.float_features,
            input.src_src_mask,
            input.tgt_tgt_mask,
        )
        # encoder_output shape: batch_size, src_seq_len, dim_model
        encoder_output = self.encode(state, src_seq, src_src_mask)

        batch_size, tgt_seq_len, _ = tgt_in_seq.shape
        # tgt_src_mask shape: batch_size, tgt_seq_len, src_seq_len
        tgt_src_mask = torch.ones(
            batch_size, tgt_seq_len, self.max_src_seq_len, device=src_src_mask.device
        )

        # decoder_output shape: batch_size, tgt_seq_len, dim_model
        decoder_output = self.decode(
            memory=encoder_output,
            state=state,
            tgt_src_mask=tgt_src_mask,
            tgt_in_seq=tgt_in_seq,
            tgt_tgt_mask=tgt_tgt_mask,
            tgt_seq_len=tgt_seq_len,
        )

        # use the decoder's last step embedding to predict the slate reward
        pred_reward = self.proj(decoder_output[:, -1, :])
        return rlt.RewardNetworkOutput(predicted_reward=pred_reward)


class Seq2SlateRewardNetJITWrapper(ModelBase):
    def __init__(self, model: Seq2SlateRewardNet):
        super().__init__()
        self.model = model

    def input_prototype(self, use_gpu=False):
        input_prototype = self.model.input_prototype()
        if use_gpu:
            input_prototype = input_prototype.cuda()
        return (
            input_prototype.state.float_features,
            input_prototype.src_seq.float_features,
            input_prototype.tgt_out_seq.float_features,
            input_prototype.src_src_mask,
            input_prototype.tgt_out_idx,
        )

    def forward(
        self,
        state: torch.Tensor,
        src_seq: torch.Tensor,
        tgt_out_seq: torch.Tensor,
        src_src_mask: torch.Tensor,
        tgt_out_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            rlt.PreprocessedRankingInput(
                state=rlt.FeatureData(float_features=state),
                src_seq=rlt.FeatureData(float_features=src_seq),
                tgt_out_seq=rlt.FeatureData(float_features=tgt_out_seq),
                src_src_mask=src_src_mask,
                tgt_out_idx=tgt_out_idx,
            )
        ).predicted_reward
