from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import numpy as np
import unittest

from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.evaluator import Evaluator
from ml.rl.thrift.core.ttypes import \
    RLParameters, TrainingParameters, DiscreteActionModelParameters
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator
from ml.rl.test.gridworld.gridworld_base import DISCOUNT


class TestGridworld(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

        self._env = Gridworld()
        self._rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.5,
            reward_burnin=10,
            maxq_learning=False
        )
        self._rl_parameters_maxq = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.5,
            reward_burnin=10,
            maxq_learning=True
        )
        self._sarsa_parameters = DiscreteActionModelParameters(
            actions=self._env.ACTIONS,
            rl=self._rl_parameters,
            training=TrainingParameters(
                layers=[self._env.num_states, 1],
                activations=['linear'],
                minibatch_size=1024,
                learning_rate=0.01,
                optimizer='ADAM',
            )
        )
        self._trainer = DiscreteActionTrainer(
            self._env.normalization, self._sarsa_parameters
        )

    def test_sarsa_layer_validation(self):
        invalid_sarsa_params = DiscreteActionModelParameters(
            actions=self._env.ACTIONS,
            rl=self._rl_parameters,
            training=TrainingParameters(
                layers=[self._env.num_states, 3],
                activations=['linear'],
                minibatch_size=32,
                learning_rate=0.1,
                optimizer='SGD',
            )
        )
        with self.assertRaises(Exception):
            # layers[-1] should be 1
            DiscreteActionTrainer(self._env.normalization, invalid_sarsa_params)

    def test_trainer_single_batch_maxq(self):
        maxq_sarsa_parameters = DiscreteActionModelParameters(
            actions=self._env.ACTIONS,
            rl=self._rl_parameters_maxq,
            training=TrainingParameters(
                layers=[self._env.num_states, 1],
                activations=['linear'],
                minibatch_size=1024,
                learning_rate=0.01,
                optimizer='ADAM',
            )
        )
        # construct the new trainer that using maxq
        maxq_trainer = DiscreteActionTrainer(
            self._env.normalization, maxq_sarsa_parameters
        )
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            self._env.generate_samples(100000, 1.0)
        predictor = maxq_trainer.predictor()
        tbp = self._env.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )
        evaluator = GridworldEvaluator(self._env, True)
        print("Pre-Training eval", evaluator.evaluate(predictor))
        self.assertGreater(evaluator.evaluate(predictor), 0.4)

        for _ in range(2):
            maxq_trainer.stream_df(tbp, None)
            evaluator.evaluate(predictor)

        print("Post-Training eval", evaluator.evaluate(predictor))
        self.assertLess(evaluator.evaluate(predictor), 0.1)

    def test_trainer_single_batch_sarsa(self):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            self._env.generate_samples(100000, 1.0)
        predictor = self._trainer.predictor()
        evaluator = GridworldEvaluator(self._env, False)
        tbp = self._env.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )

        self.assertGreater(evaluator.evaluate(predictor), 0.15)

        for _ in range(1):
            self._trainer.stream_df(tbp, None)
            evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.05)

    def test_evaluator_ground_truth(self):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, _ = \
            self._env.generate_samples(100000, 1.0)
        true_values = self._env.true_values_for_sample(states, actions, False)
        # Hijack the reward timeline to insert the ground truth
        reward_timelines = []
        for tv in true_values:
            reward_timelines.append({0: tv})
        evaluator = Evaluator(self._trainer, DISCOUNT)
        tbp = self._env.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )

        for _ in range(1):
            self._trainer.stream_df(tbp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.05)
        self.assertLess(evaluator.mc_loss[-1], 0.05)

    def test_evaluator_timeline(self):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            self._env.generate_samples(100000, 1.0)
        evaluator = Evaluator(self._trainer, DISCOUNT)

        tdp = self._env.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )
        for _ in range(1):
            self._trainer.stream_df(tdp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.2)
        self.assertLess(evaluator.mc_loss[-1], 0.2)
