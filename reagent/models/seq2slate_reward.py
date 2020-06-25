#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from reagent.torch_utils import gather


logger = logging.getLogger(__name__)


class Seq2SlateRewardNetBase(ModelBase):
    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        dim_model: int,
        num_stacked_layers: int,
        max_src_seq_len: int,
        max_tgt_seq_len: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.dim_model = dim_model
        self.num_stacked_layers = num_stacked_layers

        self.candidate_embedder = Embedder(candidate_dim, dim_model // 2)
        self.state_embedder = Embedder(state_dim, dim_model // 2)
        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len

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

    def _init_params(self):
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        def _num_of_params(model):
            return len(torch.cat([p.flatten() for p in model.parameters()]))

        logger.info(f"Num of total params: {_num_of_params(self)}, {self._get_name()}")


class Seq2SlateGRURewardNet(Seq2SlateRewardNetBase):
    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        num_stacked_layers: int,
        dim_model: int,
        max_src_seq_len: int,
        max_tgt_seq_len: int,
    ):
        super().__init__(
            state_dim,
            candidate_dim,
            dim_model,
            num_stacked_layers,
            max_src_seq_len,
            max_tgt_seq_len,
        )
        self.gru = nn.GRU(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=num_stacked_layers,
            batch_first=True,
        )
        self.end_of_seq_vec = nn.Parameter(
            torch.zeros(candidate_dim), requires_grad=True
        )
        self.proj = nn.Linear(2 * dim_model, 1)
        self._init_params()

    def _convert_seq2slate_to_reward_model_format(
        self, input: rlt.PreprocessedRankingInput
    ):
        device = next(self.parameters()).device
        # pyre-fixme[16]: Optional type has no attribute `float_features`.
        batch_size, tgt_seq_len, candidate_dim = input.tgt_out_seq.float_features.shape
        src_seq_len = input.src_seq.float_features.shape[1]
        assert self.max_tgt_seq_len == tgt_seq_len
        assert self.max_src_seq_len == src_seq_len

        # unselected_idx stores indices of items that are not included in the slate
        unselected_idx = torch.ones(batch_size, tgt_seq_len)
        unselected_idx[
            # pyre-fixme[16]: `Tensor` has no attribute `repeat_interleave`.
            torch.arange(batch_size, device=device).repeat_interleave(tgt_seq_len),
            # pyre-fixme[16]: Optional type has no attribute `flatten`.
            input.tgt_out_idx.flatten() - 2,
        ] = 0
        # shape: batch_size, (src_seq_len - tgt_seq_len)
        unselected_idx = torch.nonzero(unselected_idx, as_tuple=True)[1].reshape(
            batch_size, src_seq_len - tgt_seq_len
        )
        # shape: batch_size, (src_seq_len - tgt_seq_len), candidate_dim
        unselected_candidate_features = gather(
            input.src_seq.float_features, unselected_idx
        )
        # shape: batch_size, src_seq_len + 1, candidate_dim
        tgt_in_seq = torch.cat(
            (
                input.tgt_out_seq.float_features,
                unselected_candidate_features,
                # self.end_of_seq_vec.repeat(batch_size, 1, 1),
            ),
            dim=1,
        )

        return rlt.PreprocessedRankingInput.from_tensors(
            state=input.state.float_features,
            src_seq=input.src_seq.float_features,
            src_src_mask=input.src_src_mask,
            tgt_in_seq=tgt_in_seq,
        )

    def embed(self, state, tgt_in_seq):
        batch_size = state.shape[0]

        # candidate_embed: batch_size, src_seq_len + 1, dim_model/2
        candidate_embed = self.candidate_embedder(tgt_in_seq)
        # state_embed: batch_size, dim_model/2
        state_embed = self.state_embedder(state)
        # transform state_embed into shape: batch_size, src_seq_len, dim_model/2
        state_embed = state_embed.repeat(1, self.max_src_seq_len).reshape(
            batch_size, self.max_src_seq_len, -1
        )

        # Input at each encoder step is actually concatenation of state_embed
        # and candidate embed.
        # shape: batch_size, src_seq_len + 1, dim_model
        tgt_in_embed = torch.cat((state_embed, candidate_embed), dim=2)
        return tgt_in_embed

    def forward(self, input: rlt.PreprocessedRankingInput):
        input = self._convert_seq2slate_to_reward_model_format(input)
        state = input.state.float_features
        tgt_in_seq = input.tgt_in_seq.float_features

        # shape: batch_size, src_seq_len + 1, dim_modle
        tgt_in_embed = self.embed(state, tgt_in_seq)

        # output shape: batch_size, src_seq_len + 1, dim_model
        output, hn = self.gru(tgt_in_embed)
        # hn shape: batch_size, dim_model
        hn = hn[-1]  # top layer's hidden state

        # attention, using hidden as query, outputs as keys and values
        # shape: batch_size, src_seq_len + 1
        attn_weights = F.softmax(
            torch.bmm(
                output,
                hn.unsqueeze(2) / torch.sqrt(torch.tensor(self.candidate_dim).float()),
            ).squeeze(2),
            dim=1,
        )
        # shape: batch_size, dim_model
        context_vector = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)

        # reward prediction depends on hidden state of the last step + context vector
        # shape: batch_size, 2 * dim_model
        seq_embed = torch.cat((hn, context_vector), dim=1)

        # shape: batch_size, 1
        pred_reward = self.proj(seq_embed)
        return rlt.RewardNetworkOutput(predicted_reward=pred_reward)


class Seq2SlateTransformerRewardNet(Seq2SlateRewardNetBase):
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
        """
        super().__init__(
            state_dim,
            candidate_dim,
            dim_model,
            num_stacked_layers,
            max_src_seq_len,
            max_tgt_seq_len,
        )
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, dim_model)
        ff = PositionwiseFeedForward(dim_model, dim_feedforward)
        self.encoder = Encoder(
            EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers
        )
        self.decoder = Decoder(
            DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers
        )
        self.positional_encoding = PositionalEncoding(
            dim_model, max_len=2 * (max_src_seq_len + max_tgt_seq_len)
        )
        self.proj = nn.Linear(dim_model, 1)
        self.decoder_start_vec = nn.Parameter(
            torch.zeros(candidate_dim), requires_grad=True
        )

        self._init_params()

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
                torch.full(
                    (batch_size, 1),
                    DECODER_START_SYMBOL,
                    device=device,
                    dtype=torch.long,
                ),
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
    def __init__(self, model: Seq2SlateRewardNetBase):
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
