#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from itertools import permutations
from typing import List, Optional

import numpy as np
import reagent.core.types as rlt
import torch
import torch.nn as nn
from reagent.core.dataclasses import field
from reagent.core.parameters import Seq2SlateParameters
from reagent.core.torch_utils import gather
from reagent.models.seq2slate import BaselineNet, Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer


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


class Seq2SlateSimulationTrainer(Seq2SlateTrainer):
    """
    Seq2Slate learned with simulation data, with the action
    generated randomly and the reward computed by a reward network
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        params: Seq2SlateParameters = field(  # noqa: B008
            default_factory=Seq2SlateParameters
        ),
        baseline_net: Optional[BaselineNet] = None,
        baseline_warmup_num_batches: int = 0,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        baseline_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        policy_gradient_interval: int = 1,
        print_interval: int = 100,
        calc_cpe: bool = False,
        reward_network: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            seq2slate_net,
            params=params,
            baseline_net=baseline_net,
            baseline_warmup_num_batches=baseline_warmup_num_batches,
            policy_optimizer=policy_optimizer,
            baseline_optimizer=baseline_optimizer,
            policy_gradient_interval=policy_gradient_interval,
            print_interval=print_interval,
            calc_cpe=calc_cpe,
            reward_network=reward_network,
        )
        self.sim_param = params.simulation
        assert self.sim_param is not None
        # loaded when used
        self.reward_name_and_net = nn.ModuleDict({})
        self.MAX_DISTANCE = (
            seq2slate_net.max_src_seq_len * (seq2slate_net.max_src_seq_len - 1) / 2
        )

    @torch.no_grad()
    def _simulated_training_input(self, training_input: rlt.PreprocessedRankingInput):
        device = training_input.state.float_features.device
        # precision error may cause invalid actions
        valid_output = False
        while not valid_output:
            rank_output = self.seq2slate_net(
                training_input,
                mode=Seq2SlateMode.RANK_MODE,
                tgt_seq_len=self.seq2slate_net.max_tgt_seq_len,
                greedy=False,
            )
            model_propensities = rank_output.ranked_per_seq_probs
            model_actions_with_offset = rank_output.ranked_tgt_out_idx
            model_actions = model_actions_with_offset - 2
            if torch.all(model_actions >= 0):
                valid_output = True

        batch_size = model_actions_with_offset.shape[0]
        simulated_slate_features = gather(
            training_input.src_seq.float_features,
            # pyre-fixme[61]: `model_actions` may not be initialized here.
            model_actions,
        )

        if not self.reward_name_and_net:
            use_gpu = True if device == torch.device("cuda") else False
            self.reward_name_and_net = nn.ModuleDict(
                _load_reward_net(self.sim_param.reward_name_path, use_gpu)
            )

        sim_slate_reward = torch.zeros(batch_size, 1, device=device)
        for name, reward_net in self.reward_name_and_net.items():
            weight = self.sim_param.reward_name_weight[name]
            power = self.sim_param.reward_name_power[name]
            sr = reward_net(
                training_input.state.float_features,
                training_input.src_seq.float_features,
                simulated_slate_features,
                training_input.src_src_mask,
                model_actions_with_offset,
            ).detach()
            assert sr.ndim == 2, f"Slate reward {name} output should be 2-D tensor"
            sim_slate_reward += weight * (sr ** power)

        # guard-rail reward prediction range
        reward_clamp = self.sim_param.reward_clamp
        if reward_clamp is not None:
            sim_slate_reward = torch.clamp(
                sim_slate_reward, min=reward_clamp.clamp_min, max=reward_clamp.clamp_max
            )
        # guard-rail sequence similarity
        distance_penalty = self.sim_param.distance_penalty
        if distance_penalty is not None:
            sim_distance = (
                torch.tensor(
                    # pyre-fixme[16]: `int` has no attribute `__iter__`.
                    [swap_dist(x.tolist()) for x in model_actions],
                    device=device,
                )
                .unsqueeze(1)
                .float()
            )
            sim_slate_reward += distance_penalty * (self.MAX_DISTANCE - sim_distance)

        assert (
            len(sim_slate_reward.shape) == 2 and sim_slate_reward.shape[1] == 1
        ), f"{sim_slate_reward.shape}"

        on_policy_input = rlt.PreprocessedRankingInput.from_input(
            state=training_input.state.float_features,
            candidates=training_input.src_seq.float_features,
            device=device,
            # pyre-fixme[6]: Expected `Optional[torch.Tensor]` for 4th param but got
            #  `int`.
            # pyre-fixme[61]: `model_actions` may not be initialized here.
            action=model_actions,
            slate_reward=sim_slate_reward,
            # pyre-fixme[61]: `model_propensities` may not be initialized here.
            logged_propensities=model_propensities,
        )
        return on_policy_input

    def training_step(self, batch: rlt.PreprocessedRankingInput, batch_idx: int):
        assert type(batch) is rlt.PreprocessedRankingInput
        training_batch = self._simulated_training_input(batch)
        return super().training_step(training_batch, batch_idx)
