#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from itertools import permutations
from typing import List, Optional

import numpy as np
import reagent.types as rlt
import torch
from reagent.core.dataclasses import field
from reagent.core.tracker import observable
from reagent.models.seq2slate import (
    DECODER_START_SYMBOL,
    BaselineNet,
    Seq2SlateTransformerNet,
)
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import Seq2SlateParameters
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


def _load_reward_net(name_and_path, use_gpu):
    reward_name_and_net = {}
    for name, path in name_and_path.items():
        reward_network = torch.jit.load(path)
        if use_gpu:
            reward_network = reward_network.cuda()
        reward_name_and_net[name] = reward_network
    return reward_name_and_net


def swap_dist_in_slate(idx_):
    # Do not want to modify the original list because swap happens in place.
    idx = idx_.copy()
    swapcount = 0
    for j in range(len(idx)):
        for i in range(1, len(idx) - j):
            if idx[i - 1] > idx[i]:
                swapcount += 1
                idx[i - 1], idx[i] = idx[i], idx[i - 1]
    return swapcount


def swap_dist_out_slate(idx):
    return np.sum(x - i for i, x in enumerate(idx))


def swap_dist(idx: List[int]):
    """
    A distance which measures how many swaps the prod
    ordering needs to get to idx

    Examples:
    swap_dist([0, 1, 2, 4]) = 1
    swap_dist([0, 1, 5, 2]) = 3
    """
    assert type(idx) is list
    return swap_dist_in_slate(idx) + swap_dist_out_slate(idx)


@observable(
    train_ips_score=torch.Tensor,
    train_clamped_ips_score=torch.Tensor,
    train_baseline_loss=torch.Tensor,
    train_logged_slate_rank_probs=torch.Tensor,
    train_ips_ratio=torch.Tensor,
    train_clamped_ips_ratio=torch.Tensor,
    train_advantage=torch.Tensor,
)
class Seq2SlateSimulationTrainer(Trainer):
    """
    Seq2Slate learned with simulation data, with the action
    generated randomly and the reward computed by a reward network
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        minibatch_size: int,
        parameters: Seq2SlateParameters,
        baseline_net: Optional[BaselineNet] = None,
        baseline_warmup_num_batches: int = 0,
        use_gpu: bool = False,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        baseline_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        policy_gradient_interval: int = 1,
        print_interval: int = 100,
    ) -> None:
        self.sim_param = parameters.simulation
        assert self.sim_param is not None
        # loaded when used
        self.reward_name_and_net = {}
        self.parameters = parameters
        self.minibatch_size = minibatch_size
        self.use_gpu = use_gpu
        self.policy_gradient_interval = policy_gradient_interval
        self.print_interval = print_interval
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.permutation_index = torch.tensor(
            list(
                permutations(
                    # pyre-fixme[6]: Expected `Iterable[Variable[itertools._T]]` for
                    #  1st param but got `Tensor`.
                    torch.arange(seq2slate_net.max_src_seq_len),
                    seq2slate_net.max_tgt_seq_len,
                )
            ),
            device=self.device,
        ).long()

        if self.sim_param.distance_penalty is not None:
            assert self.sim_param.distance_penalty >= 0
            self.permutation_distance = (
                torch.tensor(
                    [swap_dist(x.tolist()) for x in self.permutation_index],
                    device=self.device,
                )
                .unsqueeze(1)
                .float()
            )
            self.MAX_DISTANCE = torch.max(self.permutation_distance)

        self.trainer = Seq2SlateTrainer(
            seq2slate_net,
            minibatch_size,
            self.parameters,
            baseline_net=baseline_net,
            baseline_warmup_num_batches=baseline_warmup_num_batches,
            use_gpu=use_gpu,
            policy_optimizer=policy_optimizer,
            baseline_optimizer=baseline_optimizer,
            policy_gradient_interval=policy_gradient_interval,
            print_interval=print_interval,
        )
        self.seq2slate_net = self.trainer.seq2slate_net
        self.baseline_net = self.trainer.baseline_net

    def warm_start_components(self):
        components = ["seq2slate_net"]
        return components

    def _simulated_training_input(
        self, training_input, sim_tgt_out_idx, sim_distance, device
    ):
        batch_size, max_tgt_seq_len = sim_tgt_out_idx.shape
        (
            _,
            max_src_seq_len,
            candidate_feat_dim,
        ) = training_input.src_seq.float_features.shape

        # candidates + padding_symbol + decoder_start_symbol
        candidate_size = max_src_seq_len + 2
        src_seq_augment = torch.zeros(
            batch_size, candidate_size, candidate_feat_dim, device=device
        )
        src_seq_augment[:, 2:, :] = training_input.src_seq.float_features

        sim_tgt_in_idx = torch.zeros_like(sim_tgt_out_idx).long()
        sim_tgt_in_idx[:, 0] = DECODER_START_SYMBOL
        sim_tgt_in_idx[:, 1:] = sim_tgt_out_idx[:, :-1]

        sim_tgt_in_seq = rlt.FeatureData(
            float_features=src_seq_augment[
                torch.arange(batch_size, device=device).repeat_interleave(
                    max_tgt_seq_len
                ),
                sim_tgt_in_idx.flatten(),
            ].view(batch_size, max_tgt_seq_len, candidate_feat_dim)
        )
        sim_tgt_out_seq = rlt.FeatureData(
            float_features=src_seq_augment[
                torch.arange(batch_size, device=device).repeat_interleave(
                    max_tgt_seq_len
                ),
                sim_tgt_out_idx.flatten(),
            ].view(batch_size, max_tgt_seq_len, candidate_feat_dim)
        )
        sim_tgt_out_probs = torch.tensor(
            [1.0 / len(self.permutation_index)], device=self.device
        ).repeat(batch_size)

        if not self.reward_name_and_net:
            self.reward_name_and_net = _load_reward_net(
                self.sim_param.reward_name_path, self.use_gpu
            )

        sim_slate_reward = torch.zeros_like(training_input.slate_reward)
        for name, reward_net in self.reward_name_and_net.items():
            weight = self.sim_param.reward_name_weight[name]
            sr = reward_net(
                training_input.state.float_features,
                training_input.src_seq.float_features,
                sim_tgt_out_seq.float_features,
                training_input.src_src_mask,
                sim_tgt_out_idx,
            ).detach()
            assert sr.ndim == 2, f"Slate reward {name} output should be 2-D tensor"
            sim_slate_reward += weight * sr

        # guard-rail reward prediction range
        reward_clamp = self.sim_param.reward_clamp
        if reward_clamp is not None:
            sim_slate_reward = torch.clamp(
                sim_slate_reward, min=reward_clamp.clamp_min, max=reward_clamp.clamp_max
            )
        # guard-rail sequence similarity
        distance_penalty = self.sim_param.distance_penalty
        if distance_penalty is not None:
            sim_slate_reward += distance_penalty * (self.MAX_DISTANCE - sim_distance)

        assert (
            len(sim_slate_reward.shape) == 2 and sim_slate_reward.shape[1] == 1
        ), f"{sim_slate_reward.shape}"

        on_policy_input = rlt.PreprocessedRankingInput(
            state=training_input.state,
            src_seq=training_input.src_seq,
            src_src_mask=training_input.src_src_mask,
            tgt_in_seq=sim_tgt_in_seq,
            tgt_out_seq=sim_tgt_out_seq,
            tgt_tgt_mask=training_input.tgt_tgt_mask,
            slate_reward=sim_slate_reward,
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
        sim_tgt_out_idx = self.permutation_index[random_indices] + 2
        if self.sim_param.distance_penalty is not None:
            sim_distance = self.permutation_distance[random_indices]
        else:
            sim_distance = None

        with torch.no_grad():
            # format data according to the new ordering
            training_input = self._simulated_training_input(
                training_input, sim_tgt_out_idx, sim_distance, self.device
            )

        return self.trainer.train(
            rlt.PreprocessedTrainingBatch(
                training_input=training_input, extras=training_batch.extras
            )
        )
