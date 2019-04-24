#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import random
import unittest

import numpy as np
import torch
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.test.constant_reward.env import Env
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.workflow.transitional import create_dqn_trainer_from_params


logger = logging.getLogger(__name__)


class TestConstantReward(unittest.TestCase):
    def _train(self, model_params, env):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        env.seed(42)
        trainer = create_dqn_trainer_from_params(model_params, env.normalization)
        logger.info("Generating constant_reward MDPs..")

        states, actions, rewards, next_states, next_actions, is_terminal, possible_actions, possible_next_actions = env.generate_samples_discrete(
            self.num_samples
        )

        logger.info("Preprocessing constant_reward MDPs..")

        for epoch in range(self.epochs):
            tdps = env.preprocess_samples_discrete(
                states,
                actions,
                rewards,
                next_states,
                next_actions,
                is_terminal,
                possible_actions,
                possible_next_actions,
                model_params.training.minibatch_size,
            )
            logger.info("Training.. " + str(epoch))
            for tdp in tdps:
                trainer.train(tdp)
            logger.info(
                " ".join(
                    [
                        "Training epoch",
                        str(epoch),
                        "average q values",
                        str(torch.mean(trainer.all_action_scores)),
                    ]
                )
            )
        return trainer

    def setUp(self):
        self.layers = [-1, 128, -1]
        self.activations = ["relu", "linear"]
        self.state_dims = 5
        self.action_dims = 2
        self.num_samples = 1024 * 10  # multiple of minibatch_size (1024)
        self.epochs = 24
        super().setUp()

    def test_trainer_maxq(self):
        env = Env(self.state_dims, self.action_dims)
        maxq_parameters = DiscreteActionModelParameters(
            actions=env.actions,
            rl=RLParameters(gamma=0.95, target_update_rate=0.9, maxq_learning=True),
            rainbow=RainbowDQNParameters(
                double_q_learning=True, dueling_architecture=False
            ),
            training=TrainingParameters(
                layers=self.layers,
                activations=self.activations,
                minibatch_size=1024,
                learning_rate=0.25,
                optimizer="ADAM",
            ),
        )

        # Q value should converge to very close to 20
        trainer = self._train(maxq_parameters, env)
        avg_q_value_after_training = torch.mean(trainer.all_action_scores)
        self.assertLess(avg_q_value_after_training, 22)
        self.assertGreater(avg_q_value_after_training, 18)

    def test_minibatches_per_step(self):
        _epochs = self.epochs
        self.epochs = 2
        rl_parameters = RLParameters(
            gamma=0.95, target_update_rate=0.9, maxq_learning=True
        )
        rainbow_parameters = RainbowDQNParameters(
            double_q_learning=True, dueling_architecture=False
        )
        training_parameters1 = TrainingParameters(
            layers=self.layers,
            activations=self.activations,
            minibatch_size=1024,
            minibatches_per_step=1,
            learning_rate=0.25,
            optimizer="ADAM",
        )
        training_parameters2 = TrainingParameters(
            layers=self.layers,
            activations=self.activations,
            minibatch_size=128,
            minibatches_per_step=8,
            learning_rate=0.25,
            optimizer="ADAM",
        )
        env1 = Env(self.state_dims, self.action_dims)
        env2 = Env(self.state_dims, self.action_dims)
        model_parameters1 = DiscreteActionModelParameters(
            actions=env1.actions,
            rl=rl_parameters,
            rainbow=rainbow_parameters,
            training=training_parameters1,
        )
        model_parameters2 = DiscreteActionModelParameters(
            actions=env2.actions,
            rl=rl_parameters,
            rainbow=rainbow_parameters,
            training=training_parameters2,
        )
        # minibatch_size / 8, minibatches_per_step * 8 should give the same result
        logger.info("Training model 1")
        trainer1 = self._train(model_parameters1, env1)
        SummaryWriterContext._reset_globals()
        logger.info("Training model 2")
        trainer2 = self._train(model_parameters2, env2)

        weight1 = trainer1.q_network.fc.layers[-1].weight.detach().numpy()
        weight2 = trainer2.q_network.fc.layers[-1].weight.detach().numpy()

        # Due to numerical stability this tolerance has to be fairly high
        self.assertTrue(np.allclose(weight1, weight2, rtol=0.0, atol=1e-3))
        self.epochs = _epochs
