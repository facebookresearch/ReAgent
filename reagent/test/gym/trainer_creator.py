#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import torch
from reagent.models.actor import GaussianFullyConnectedActor
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.models.parametric_dqn import FullyConnectedParametricDQN
from reagent.parameters import FeedForwardParameters, RLParameters
from reagent.preprocessing.normalization import get_num_output_features
from reagent.test.gym.open_ai_gym_environment import OpenAIGymEnvironment
from reagent.training.sac_trainer import SACTrainer, SACTrainerParameters


def get_sac_trainer(
    env: OpenAIGymEnvironment,
    rl_parameters: RLParameters,
    trainer_parameters: SACTrainerParameters,
    critic_training: FeedForwardParameters,
    actor_training: FeedForwardParameters,
    sac_value_training: Optional[FeedForwardParameters],
    use_gpu: bool,
) -> SACTrainer:
    assert rl_parameters == trainer_parameters.rl
    state_dim = get_num_output_features(env.normalization)
    action_dim = get_num_output_features(env.normalization_action)
    q1_network = FullyConnectedParametricDQN(
        state_dim, action_dim, critic_training.layers, critic_training.activations
    )
    q2_network = None
    # TODO:
    # if trainer_parameters.use_2_q_functions:
    #     q2_network = FullyConnectedParametricDQN(
    #         state_dim,
    #         action_dim,
    #         critic_training.layers,
    #         critic_training.activations,
    #     )
    value_network = None
    if sac_value_training:
        value_network = FullyConnectedNetwork(
            [state_dim] + sac_value_training.layers + [1],
            sac_value_training.activations + ["linear"],
        )
    actor_network = GaussianFullyConnectedActor(
        state_dim, action_dim, actor_training.layers, actor_training.activations
    )

    min_action_range_tensor_training = torch.full((1, action_dim), -1 + 1e-6)
    max_action_range_tensor_training = torch.full((1, action_dim), 1 - 1e-6)
    min_action_range_tensor_serving = (
        torch.from_numpy(env.action_space.low).float().unsqueeze(dim=0)  # type: ignore
    )
    max_action_range_tensor_serving = (
        torch.from_numpy(env.action_space.high).float().unsqueeze(dim=0)  # type: ignore
    )

    if use_gpu:
        q1_network.cuda()
        if q2_network:
            q2_network.cuda()
        if value_network:
            value_network.cuda()
        actor_network.cuda()

        min_action_range_tensor_training = min_action_range_tensor_training.cuda()
        max_action_range_tensor_training = max_action_range_tensor_training.cuda()
        min_action_range_tensor_serving = min_action_range_tensor_serving.cuda()
        max_action_range_tensor_serving = max_action_range_tensor_serving.cuda()

    return SACTrainer(
        q1_network,
        actor_network,
        trainer_parameters,
        use_gpu=use_gpu,
        value_network=value_network,
        q2_network=q2_network,
        min_action_range_tensor_training=min_action_range_tensor_training,
        max_action_range_tensor_training=max_action_range_tensor_training,
        min_action_range_tensor_serving=min_action_range_tensor_serving,
        max_action_range_tensor_serving=max_action_range_tensor_serving,
    )
