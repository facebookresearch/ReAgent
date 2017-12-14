from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import numpy as np
import unittest

from libfb.py.testutil import data_provider

from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.evaluator import Evaluator
from ml.rl.thrift.core.ttypes import \
    RLParameters, TrainingParameters, DiscreteActionModelParameters
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_enum import GridworldEnum
from ml.rl.test.gridworld.gridworld_evaluator import GridworldEvaluator, \
    GridworldEnumEvaluator
from ml.rl.test.gridworld.gridworld_base import DISCOUNT


class DataProvider(object):
    @staticmethod
    def envs():
        return [
            (Gridworld(),),
            (GridworldEnum(),)
        ]

    @staticmethod
    def envs_and_evaluators():
        return [
            (
                Gridworld(),
                GridworldEvaluator
            ),
            (
                GridworldEnum(),
                GridworldEnumEvaluator
            ),
        ]


class TestGridworld(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

    def get_sarsa_trainer(self, environment):
        rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.5,
            reward_burnin=10,
            maxq_learning=False
        )
        training_parameters = TrainingParameters(
            layers=[-1, 1],
            activations=['linear'],
            minibatch_size=1024,
            learning_rate=0.01,
            optimizer='ADAM',
        )
        return DiscreteActionTrainer(
            environment.normalization, DiscreteActionModelParameters(
                actions=environment.ACTIONS,
                rl=rl_parameters,
                training=training_parameters
            )
        )

    def test_sarsa_layer_validation(self):
        env = Gridworld()
        invalid_sarsa_params = DiscreteActionModelParameters(
            actions=env.ACTIONS,
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=0.5,
                reward_burnin=10,
                maxq_learning=False
            ),
            training=TrainingParameters(
                layers=[-1, 3],
                activations=['linear'],
                minibatch_size=32,
                learning_rate=0.1,
                optimizer='SGD',
            )
        )
        with self.assertRaises(Exception):
            # layers[-1] should be 1
            DiscreteActionTrainer(env.normalization, invalid_sarsa_params)

    @data_provider(DataProvider.envs_and_evaluators, new_fixture=True)
    def test_trainer_single_batch_maxq(self, environment, evaluator_class):
        maxq_sarsa_parameters = DiscreteActionModelParameters(
            actions=environment.ACTIONS,
            rl=RLParameters(
                gamma=DISCOUNT,
                target_update_rate=0.5,
                reward_burnin=10,
                maxq_learning=True
            ),
            training=TrainingParameters(
                layers=[-1, 1],
                activations=['linear'],
                minibatch_size=1024,
                learning_rate=0.01,
                optimizer='ADAM',
            )
        )
        # construct the new trainer that using maxq
        maxq_trainer = DiscreteActionTrainer(
            environment.normalization, maxq_sarsa_parameters
        )
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            environment.generate_samples(100000, 1.0)
        predictor = maxq_trainer.predictor()
        tdp = environment.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )
        evaluator = evaluator_class(environment, True)
        print("Pre-Training eval", evaluator.evaluate(predictor))
        self.assertGreater(evaluator.evaluate(predictor), 0.4)

        for _ in range(2):
            maxq_trainer.stream_tdp(tdp, None)
            evaluator.evaluate(predictor)

        print("Post-Training eval", evaluator.evaluate(predictor))
        self.assertLess(evaluator.evaluate(predictor), 0.1)

    @data_provider(DataProvider.envs_and_evaluators, new_fixture=True)
    def test_trainer_single_batch_sarsa(self, environment, evaluator_class):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            environment.generate_samples(100000, 1.0)
        evaluator = evaluator_class(environment, False)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        tdp = environment.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )

        self.assertGreater(evaluator.evaluate(predictor), 0.15)

        trainer.stream_tdp(tdp, None)
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.05)

    @data_provider(DataProvider.envs_and_evaluators, new_fixture=True)
    def test_trainer_many_batch_sarsa(self, environment, evaluator_class):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            environment.generate_samples(100000, 1.0)
        trainer = self.get_sarsa_trainer(environment)
        predictor = trainer.predictor()
        evaluator = evaluator_class(environment, False)
        tdp = environment.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )

        print("Pre-Training eval", evaluator.evaluate(predictor))
        self.assertGreater(evaluator.evaluate(predictor), 0.15)

        for i in range(0, tdp.size(), 10):
            trainer.stream_tdp(tdp.get_sub_page(i, i + 10), None)

        print("Post-Training eval", evaluator.evaluate(predictor))
        evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.05)

    @data_provider(DataProvider.envs, new_fixture=True)
    def test_evaluator_ground_truth(self, environment):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, _ = environment.generate_samples(100000, 1.0)
        true_values = environment.true_values_for_sample(states, actions, False)
        # Hijack the reward timeline to insert the ground truth
        reward_timelines = []
        for tv in true_values:
            reward_timelines.append({0: tv})
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator(trainer, DISCOUNT)
        tdp = environment.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )

        trainer.stream_tdp(tdp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.05)
        self.assertLess(evaluator.mc_loss[-1], 0.05)

    @data_provider(DataProvider.envs, new_fixture=True)
    def test_evaluator_timeline(self, environment):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            environment.generate_samples(100000, 1.0)
        trainer = self.get_sarsa_trainer(environment)
        evaluator = Evaluator(trainer, DISCOUNT)

        tdp = environment.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )
        trainer.stream_tdp(tdp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.2)
        self.assertLess(evaluator.mc_loss[-1], 0.2)
