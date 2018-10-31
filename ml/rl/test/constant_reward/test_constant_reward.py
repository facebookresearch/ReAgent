#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import torch
from ml.rl.test.constant_reward.env import Env
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.dqn_trainer import DQNTrainer


logger = logging.getLogger(__name__)


class TestConstantReward(unittest.TestCase):
    def setUp(self):
        self.layers = [-1, 128, -1]
        self.activations = ["relu", "linear"]
        self.state_dims = 5
        self.action_dims = 2
        self.num_samples = 10000
        self.minibatch_size = 128
        self.epochs = 50
        super(self.__class__, self).setUp()

    def test_trainer_maxq(self):
        env = Env(self.state_dims, self.action_dims)
        env.seed(42)
        maxq_parameters = DiscreteActionModelParameters(
            actions=env.actions,
            rl=RLParameters(
                gamma=0.99,
                target_update_rate=1.0,
                reward_burnin=100,
                maxq_learning=True,
            ),
            rainbow=RainbowDQNParameters(
                double_q_learning=True, dueling_architecture=False
            ),
            training=TrainingParameters(
                layers=self.layers,
                activations=self.activations,
                minibatch_size=self.minibatch_size,
                learning_rate=1.0,
                optimizer="ADAM",
            ),
        )
        maxq_trainer = DQNTrainer(maxq_parameters, env.normalization)

        logger.info("Generating constant_reward MDPs..")

        states, actions, rewards, next_states, next_actions, is_terminal, possible_next_actions = env.generate_samples_discrete(
            self.num_samples
        )

        logger.info("Preprocessing constant_reward MDPs..")

        tdps = env.preprocess_samples_discrete(
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            is_terminal,
            possible_next_actions,
            self.minibatch_size,
        )

        for epoch in range(self.epochs):
            logger.info("Training.. " + str(epoch))
            for tdp in tdps:
                maxq_trainer.train(tdp, None)
            logger.info(
                " ".join(
                    [
                        "Training epoch",
                        str(epoch),
                        "average q values",
                        str(torch.mean(maxq_trainer.all_action_scores)),
                    ]
                )
            )

        # Q value should converge to very close to 100
        avg_q_value_after_training = torch.mean(maxq_trainer.all_action_scores)

        self.assertLess(avg_q_value_after_training, 101)
        self.assertGreater(avg_q_value_after_training, 99)
