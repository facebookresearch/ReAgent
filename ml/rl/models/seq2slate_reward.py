#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy

import torch
import torch.nn as nn
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.seq2slate import (
    Decoder,
    DecoderLayer,
    Embedder,
    Encoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
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
        self.slate_seq_len = max_tgt_seq_len

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
            dim_model, max_len=2 * self.slate_seq_len
        )
        self.decoder_embedder = nn.Linear(candidate_dim, dim_model)
        self.proj = nn.Linear(dim_model, 1)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _convert_seq2slate_to_reward_model_format(
        self, input: rlt.PreprocessedRankingInput
    ):
        """
        Different than seq2slate models, the reward model applies the encoder
        on slate data (target sequences in seq2slate data). So we need to
        re-assemble data: src_seq, tgt_seq and masks should change
        correspondingly.
        """
        device = next(self.parameters()).device
        batch_size, _, candidate_dim = input.src_seq.float_features.shape
        assert input.tgt_out_idx is not None
        slate_seq_len = input.tgt_out_idx.shape[1]
        new_src_seq = input.src_seq.float_features[
            torch.arange(batch_size).repeat_interleave(slate_seq_len),  # type: ignore
            # -2 to offset decoder starting symbol and padding symbol,
            input.tgt_out_idx.flatten() - 2,
        ].reshape(batch_size, slate_seq_len, candidate_dim)

        return rlt.PreprocessedRankingInput.from_tensors(
            state=input.state.float_features,
            src_seq=new_src_seq,
            src_src_mask=torch.ones(batch_size, slate_seq_len, slate_seq_len).to(
                device
            ),
            tgt_seq=torch.ones(batch_size, 1, candidate_dim).to(device),
            tgt_tgt_mask=torch.ones(batch_size, 1, 1).to(device),
            slate_reward=input.slate_reward,
        )

    def encode(self, state, src_seq, src_mask):
        # state: batch_size, state_dim
        # src_seq: batch_size, slate_seq_len, dim_candidate
        # src_src_mask shape: batch_size, slate_seq_len, slate_seq_len
        batch_size = src_seq.shape[0]

        # candidate_embed: batch_size, slate_seq_len, dim_model/2
        candidate_embed = self.candidate_embedder(src_seq)
        # state_embed: batch_size, dim_model/2
        state_embed = self.state_embedder(state)
        # transform state_embed into shape: batch_size, slate_seq_len, dim_model/2
        state_embed = state_embed.repeat(1, self.slate_seq_len).reshape(
            batch_size, self.slate_seq_len, -1
        )

        # Input at each encoder step is actually concatenation of state_embed
        # and candidate embed. state_embed is replicated at each encoding step.
        # src_embed shape: batch_size, slate_seq_len, dim_model
        src_embed = self.positional_encoding(
            torch.cat((state_embed, candidate_embed), dim=2), self.slate_seq_len
        )

        # encoder_output shape: batch_size, slate_seq_len, dim_model
        return self.encoder(src_embed, src_mask)

    def decode(self, memory, state, tgt_src_mask, tgt_seq, tgt_tgt_mask):
        """
        One step decoder. The decoder's output will be used as the input to
        the last layer for predicting slate reward
        """
        # tgt_embed shape: batch_size, 1, dim_model
        tgt_embed = self.decoder_embedder(tgt_seq)
        # shape: batch_size, 1, dim_model
        return self.decoder(tgt_embed, memory, tgt_src_mask, tgt_tgt_mask)

    def input_prototype(self):
        return rlt.PreprocessedRankingInput.from_tensors(
            state=torch.randn(1, self.state_dim),
            src_seq=torch.randn(1, self.max_src_seq_len, self.candidate_dim),
            tgt_seq=torch.randn(1, self.max_tgt_seq_len, self.candidate_dim),
            src_src_mask=torch.ones(1, self.max_src_seq_len, self.max_src_seq_len),
            tgt_tgt_mask=torch.ones(1, self.max_tgt_seq_len, self.max_tgt_seq_len),
            slate_reward=torch.randn(1),
            tgt_out_idx=torch.arange(self.max_tgt_seq_len).reshape(1, -1) + 2,
        )

    def forward(self, input: rlt.PreprocessedRankingInput):
        """ Encode tgt sequences and predict the slate reward. """
        input = self._convert_seq2slate_to_reward_model_format(input)
        assert input.tgt_seq is not None
        state, src_seq, tgt_seq, src_src_mask, tgt_tgt_mask = (
            input.state.float_features,
            input.src_seq.float_features,
            input.tgt_seq.float_features,
            input.src_src_mask,
            input.tgt_tgt_mask,
        )
        # tgt_src_mask shape: batch_size, 1, slate_seq_len
        tgt_src_mask = src_src_mask[:, :1, :]

        memory = self.encode(state, src_seq, src_src_mask)
        # out shape: batch_size, 1, dim_model
        out = self.decode(
            memory=memory,
            state=state,
            tgt_src_mask=tgt_src_mask,
            tgt_seq=tgt_seq,
            tgt_tgt_mask=tgt_tgt_mask,
        )
        out = out.squeeze(1)

        pred_reward = self.proj(out).squeeze()
        return rlt.RewardNetworkOutput(predicted_reward=pred_reward)


class Seq2SlateRewardNetJITWrapper(ModelBase):
    def __init__(self, model: Seq2SlateRewardNet):
        super().__init__()
        self.model = model

    def input_prototype(self):
        input_prototype = self.model.input_prototype()
        return (
            input_prototype.state.float_features,
            input_prototype.src_seq.float_features,
            input_prototype.src_src_mask,
            input_prototype.slate_reward,
            input_prototype.tgt_out_idx,
        )

    def forward(
        self,
        state: torch.Tensor,
        src_seq: torch.Tensor,
        src_src_mask: torch.Tensor,
        slate_reward: torch.Tensor,
        tgt_out_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            rlt.PreprocessedRankingInput(
                state=rlt.PreprocessedFeatureVector(float_features=state),
                src_seq=rlt.PreprocessedFeatureVector(float_features=src_seq),
                src_src_mask=src_src_mask,
                slate_reward=slate_reward,
                tgt_out_idx=tgt_out_idx,
            )
        ).predicted_reward
