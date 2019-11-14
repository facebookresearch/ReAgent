#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from functools import partial
from typing import Tuple

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.simulators.recsim import DocumentFeature, RecSim
from ml.rl.test.base.horizon_test_base import HorizonTestBase
from ml.rl.test.gym.open_ai_gym_environment import ModelType
from ml.rl.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool
from ml.rl.training.slate_q_trainer import SlateQTrainer, SlateQTrainerParameters


def _random_policy(
    obs: Tuple[torch.Tensor, torch.Tensor, DocumentFeature], recsim: RecSim
):
    active_user_idxs, user_features, candidate_features = obs
    item_idxs = torch.multinomial(
        torch.ones(active_user_idxs.shape[0], recsim.m), recsim.k
    )
    return item_idxs


def _top_k_policy(
    q_network, obs: Tuple[torch.Tensor, torch.Tensor, DocumentFeature], recsim: RecSim
):
    active_user_idxs, user_features, candidate_features = obs

    slate_with_null = recsim.select(
        candidate_features,
        torch.repeat_interleave(
            torch.arange(recsim.m).unsqueeze(dim=0), active_user_idxs.shape[0], dim=0
        ),
        add_null=True,
    )
    _user_choice, interest = recsim.compute_user_choice(slate_with_null)
    propensity = F.softmax(interest, dim=1)[:, : recsim.m]

    tiled_user_features = torch.repeat_interleave(user_features, recsim.m, dim=0)
    candidate_feature_vector = candidate_features.as_vector()
    action_dim = candidate_feature_vector.shape[2]
    flatten_candidate_features = candidate_feature_vector.view(-1, action_dim)

    q_network_input = rlt.PreprocessedStateAction.from_tensors(
        state=tiled_user_features, action=flatten_candidate_features
    )
    q_values = q_network(q_network_input).q_value.view(-1, recsim.m)

    values = q_values * propensity

    top_values, item_idxs = torch.topk(values, recsim.k, dim=1)
    return item_idxs


class TestSlateQTrainer(HorizonTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        np.random.seed(0)

    def test_slate_q_trainer(self):
        recsim = RecSim(num_users=10)

        # Build memory pool with random policy
        memory_pool = OpenAIGymMemoryPool(10000000)
        random_reward = recsim.rollout_policy(_random_policy, memory_pool)

        # Train a model
        q_network = FullyConnectedParametricDQN(
            state_dim=memory_pool.state_dim,
            action_dim=memory_pool.action_dim,
            sizes=[64, 32],
            activations=["relu", "relu"],
        )

        q_network = q_network.eval()
        recsim.reset()
        untrained_policy_reward = recsim.rollout_policy(
            partial(_top_k_policy, q_network)
        )
        q_network = q_network.train()

        q_network_target = q_network.get_target_network()
        parameters = SlateQTrainerParameters()
        trainer = SlateQTrainer(q_network, q_network_target, parameters)

        for _i in range(1000):
            tdp = memory_pool.sample_memories(
                128, model_type=ModelType.PYTORCH_PARAMETRIC_DQN.value
            )
            training_batch = tdp.as_slate_q_training_batch()
            trainer.train(training_batch)

        q_network = q_network.eval()
        recsim.reset()
        trained_policy_reward = recsim.rollout_policy(partial(_top_k_policy, q_network))

        print(
            f"Reward; random: {random_reward}; untrained: {untrained_policy_reward}; "
            f"trained: {trained_policy_reward}"
        )

        self.assertGreater(trained_policy_reward, untrained_policy_reward)
        self.assertGreater(trained_policy_reward, random_reward)
        self.assertEqual(random_reward, 1384.0)
        self.assertEqual(untrained_policy_reward, 1200.0)
        self.assertEqual(trained_policy_reward, 1432.0)
