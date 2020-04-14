#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from itertools import permutations
from typing import Optional

import reagent.types as rlt
import torch
from reagent.models.seq2slate import (
    DECODER_START_SYMBOL,
    BaselineNet,
    Seq2SlateTransformerNet,
)
from reagent.parameters import Seq2SlateTransformerParameters
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


def _load_reward_net(path, use_gpu):
    reward_network = torch.jit.load(path)
    if use_gpu:
        reward_network = reward_network.cuda()
    return reward_network


class Seq2SlateSimulationTrainer(Trainer):
    """
    Seq2Slate learned with simulation data, with the action
    generated randomly and the reward computed by a reward network
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        parameters: Seq2SlateTransformerParameters,
        minibatch_size: int,
        reward_net_path: str,
        baseline_net: Optional[BaselineNet] = None,
        use_gpu: bool = False,
    ) -> None:
        self.reward_net_path = reward_net_path
        # loaded when used
        self.reward_net = None
        self.parameters = parameters
        self.minibatch_size = minibatch_size
        self.use_gpu = use_gpu
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.permutation_index = torch.tensor(
            list(
                permutations(
                    torch.arange(seq2slate_net.max_src_seq_len),
                    seq2slate_net.max_tgt_seq_len,
                )
            ),
            device=self.device,
        ).long()
        self.trainer = Seq2SlateTrainer(
            seq2slate_net, parameters, minibatch_size, baseline_net, use_gpu
        )
        self.seq2slate_net = self.trainer.seq2slate_net
        self.baseline_net = self.trainer.baseline_net

    def warm_start_components(self):
        components = ["seq2slate_net"]
        return components

    def _simulated_training_input(self, training_input, sim_tgt_out_idx, device):
        batch_size, max_tgt_seq_len = sim_tgt_out_idx.shape
        _, max_src_seq_len, candidate_feat_dim = (
            training_input.src_seq.float_features.shape
        )

        # candidates + padding_symbol + decoder_start_symbol
        candidate_size = max_src_seq_len + 2
        src_seq_augment = torch.zeros(
            batch_size, candidate_size, candidate_feat_dim, device=device
        )
        src_seq_augment[:, 2:, :] = training_input.src_seq.float_features

        sim_tgt_in_idx = torch.zeros_like(sim_tgt_out_idx).long()
        sim_tgt_in_idx[:, 0] = DECODER_START_SYMBOL
        sim_tgt_in_idx[:, 1:] = sim_tgt_out_idx[:, :-1]

        sim_tgt_in_seq = rlt.PreprocessedFeatureVector(
            float_features=src_seq_augment[
                torch.arange(
                    batch_size, device=device
                ).repeat_interleave(  # type: ignore
                    max_tgt_seq_len
                ),
                sim_tgt_in_idx.flatten(),
            ].view(batch_size, max_tgt_seq_len, -1)
        )
        sim_tgt_out_seq = rlt.PreprocessedFeatureVector(
            float_features=src_seq_augment[
                torch.arange(
                    batch_size, device=device
                ).repeat_interleave(  # type: ignore
                    max_tgt_seq_len
                ),
                sim_tgt_out_idx.flatten(),
            ].view(batch_size, max_tgt_seq_len, -1)
        )
        sim_tgt_out_probs = torch.tensor(
            [1.0 / len(self.permutation_index)], device=self.device
        ).repeat(batch_size)

        if self.reward_net is None:
            self.reward_net = _load_reward_net(self.reward_net_path, self.use_gpu)
        slate_reward = (
            self.reward_net(
                training_input.state.float_features,
                training_input.src_seq.float_features,
                sim_tgt_out_seq.float_features,
                training_input.src_src_mask,
                # TODO: reward_network should not need slate_reward as input
                training_input.slate_reward,
                sim_tgt_out_idx,
            )
            .squeeze()
            .detach()
        )
        # guard-rail reward prediction
        reward_clamp = self.parameters.simulation_reward_clamp
        if reward_clamp is not None:
            slate_reward = torch.clamp(
                slate_reward, min=reward_clamp.clamp_min, max=reward_clamp.clamp_max
            )

        on_policy_input = rlt.PreprocessedRankingInput(
            state=training_input.state,
            src_seq=training_input.src_seq,
            src_src_mask=training_input.src_src_mask,
            tgt_in_seq=sim_tgt_in_seq,
            tgt_out_seq=sim_tgt_out_seq,
            tgt_tgt_mask=training_input.tgt_tgt_mask,
            slate_reward=slate_reward,
            src_in_idx=training_input.src_in_idx,
            tgt_in_idx=sim_tgt_in_idx,
            tgt_out_idx=sim_tgt_out_idx,
            tgt_out_probs=sim_tgt_out_probs,
        )
        return on_policy_input

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        assert type(training_batch) is rlt.PreprocessedTrainingBatch
        training_input = training_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedRankingInput)

        batch_size = training_input.state.float_features.shape[0]

        # randomly pick a permutation for every slate
        random_indices = torch.randint(0, len(self.permutation_index), (batch_size,))
        random_tgt_out_idx = self.permutation_index[random_indices] + 2
        with torch.no_grad():
            # then format data according to the new ordering
            training_input = self._simulated_training_input(
                training_input, random_tgt_out_idx, self.device
            )

        return self.trainer.train(
            rlt.PreprocessedTrainingBatch(
                training_input=training_input, extras=training_batch.extras
            )
        )
