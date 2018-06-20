#!/usr/bin/env python3

import numpy as np
import unittest

from caffe2.python import workspace
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.thrift.core.ttypes import \
    RLParameters, TrainingParameters, DiscreteActionModelParameters
from ml.rl.test.constant_reward.env import Env

import logging

logger = logging.getLogger(__name__)


class TestConstantReward(unittest.TestCase):
    def setUp(self):
        self.layers = [-1, 512, 256, 128, 2]
        self.activations = ['relu', 'relu', 'relu', 'linear']
        self.state_dims = 13
        self.action_dims = 10
        self.num_samples = 100000
        self.minibatch_size = 256
        self.epochs = 5
        super(self.__class__, self).setUp()

    def test_trainer_maxq(self):
        env = Env(self.state_dims, self.action_dims)
        env.seed(42)
        maxq_parameters = DiscreteActionModelParameters(
            actions=env.actions,
            rl=RLParameters(
                gamma=0.99,
                target_update_rate=0.5,
                reward_burnin=10,
                maxq_learning=True
            ),
            training=TrainingParameters(
                layers=self.layers,
                activations=self.activations,
                minibatch_size=self.minibatch_size,
                learning_rate=0.01,
                optimizer='ADAM',
            )
        )
        maxq_trainer = DiscreteActionTrainer(
            maxq_parameters,
            env.normalization
        )
        # predictor = maxq_trainer.predictor()

        logger.info('Generating constant_reward MDPs..')

        states, actions, rewards, next_states, next_actions, is_terminal, \
            possible_next_actions, reward_timelines = \
            env.generate_samples_discrete(self.num_samples)

        logger.info('Preprocessing constant_reward MDPs..')

        tdps = env.preprocess_samples_discrete(states, actions, rewards, next_states,
            next_actions, is_terminal, possible_next_actions, reward_timelines,
            self.minibatch_size,)

        for epoch in range(self.epochs):
            logger.info('Training..', epoch)
            for tdp in tdps:
                maxq_trainer.train_numpy(tdp, None)
            logger.info('Training epoch', epoch, 'average q values',
                  np.mean(workspace.FetchBlob(maxq_trainer.q_score_output)),
                  'td_loss', workspace.FetchBlob(maxq_trainer.loss_blob))

        # Q value should converge to very close to 100
        avg_q_value_after_training = np.mean(
            workspace.FetchBlob(maxq_trainer.q_score_output))

        self.assertLess(avg_q_value_after_training, 101)
        self.assertGreater(avg_q_value_after_training, 99)
