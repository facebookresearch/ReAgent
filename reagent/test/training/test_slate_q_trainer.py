#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from functools import partial
from typing import Tuple

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.simulators.recsim import RecSim, random_policy, top_k_policy
from ml.rl.test.base.horizon_test_base import HorizonTestBase
from ml.rl.test.gym.open_ai_gym_environment import ModelType
from ml.rl.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool
from ml.rl.training.slate_q_trainer import SlateQTrainer, SlateQTrainerParameters


class TestSlateQTrainer(HorizonTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        np.random.seed(0)

    def test_slate_q_trainer(self):
        recsim = RecSim(num_users=10)

        # Build memory pool with random policy
        memory_pool = OpenAIGymMemoryPool(10000000)
        random_reward = recsim.rollout_policy(random_policy, memory_pool)

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
            partial(top_k_policy, q_network)
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
        trained_policy_reward = recsim.rollout_policy(partial(top_k_policy, q_network))

        print(
            f"Reward; random: {random_reward}; untrained: {untrained_policy_reward}; "
            f"trained: {trained_policy_reward}"
        )

        self.assertGreater(trained_policy_reward, untrained_policy_reward)
        self.assertGreater(trained_policy_reward, random_reward)
        self.assertEqual(random_reward, 1384.0)
        self.assertEqual(untrained_policy_reward, 1200.0)
        self.assertEqual(trained_policy_reward, 1432.0)
