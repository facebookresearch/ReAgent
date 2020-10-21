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
from reagent.models.seq2slate import BaselineNet, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import Seq2SlateParameters
from reagent.torch_utils import gather
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
        self, training_input, simulation_action, sim_distance
    ):
        batch_size, max_tgt_seq_len = simulation_action.shape
        simulate_slate_features = rlt.FeatureData(
            float_features=gather(
                training_input.src_seq.float_features, simulation_action
            )
        )
        simulation_sample_propensities = torch.tensor(
            [1.0 / len(self.permutation_index)], device=self.device
        ).repeat(batch_size, 1)

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
                simulate_slate_features.float_features,
                training_input.src_src_mask,
                simulation_action + 2,  # offset by 2 reserved symbols
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

        on_policy_input = rlt.PreprocessedRankingInput.from_input(
            state=training_input.state.float_features,
            candidates=training_input.src_seq.float_features,
            device=self.device,
            action=simulation_action,
            slate_reward=sim_slate_reward,
            logged_propensities=simulation_sample_propensities,
        )

        return on_policy_input

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        assert type(training_batch) is rlt.PreprocessedTrainingBatch
        training_input = training_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedRankingInput)

        batch_size = training_input.state.float_features.shape[0]

        # randomly pick a permutation for every slate
        random_indices = torch.randint(0, len(self.permutation_index), (batch_size,))
        simulation_action = self.permutation_index[random_indices]
        if self.sim_param.distance_penalty is not None:
            sim_distance = self.permutation_distance[random_indices]
        else:
            sim_distance = None

        with torch.no_grad():
            # format data according to the new ordering
            training_input = self._simulated_training_input(
                training_input, simulation_action, sim_distance
            )

        return self.trainer.train(
            rlt.PreprocessedTrainingBatch(
                training_input=training_input, extras=training_batch.extras
            )
        )
