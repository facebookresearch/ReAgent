from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# @build:deps [
# @/caffe2/caffe2/python:caffe2_py
# @/caffe2/caffe2/fb/data:hive_reader_python
# @/caffe2/proto:fb_protobuf
# @/hiveio:par_init
# ]

import random
import numpy as np
import unittest

from ml.rl.training.evaluator import \
    Evaluator
from ml.rl.thrift.core.ttypes import \
    RLParameters, TrainingParameters, \
    ContinuousActionModelParameters, KnnParameters
from ml.rl.training.continuous_action_dqn_trainer import \
    ContinuousActionDQNTrainer
from ml.rl.test.gridworld.gridworld_base import \
    DISCOUNT
from ml.rl.test.gridworld.gridworld_continuous import \
    GridworldContinuous
from ml.rl.test.gridworld.gridworld_evaluator import \
    GridworldContinuousEvaluator


class TestGridworldContinuous(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

        self._env = GridworldContinuous()
        self._rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.5,
            reward_burnin=10,
            maxq_learning=False,
        )
        self._rl_parameters_maxq = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.5,
            reward_burnin=10,
            maxq_learning=True,
        )
        self._rl_parameters = ContinuousActionModelParameters(
            rl=self._rl_parameters,
            training=TrainingParameters(
                layers=[
                    -1, self._env.num_states * self._env.num_actions * 2, 1
                ],
                activations=['linear', 'linear'],
                minibatch_size=1024,
                learning_rate=0.01,
                optimizer='ADAM',
            ),
            knn=KnnParameters(
                model_type='DQN',
                # model_type='KNN_DRRN', model_knn_freq=100, model_knn_k=1
            )
        )
        self._trainer = ContinuousActionDQNTrainer(
            self._env.normalization, self._env.normalization_action,
            self._rl_parameters
        )

    def test_trainer_single_batch_sarsa(self):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            self._env.generate_samples(100000, 1.0)
        predictor = self._trainer.predictor()
        evaluator = GridworldContinuousEvaluator(self._env, False)
        tbp = self._env.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )

        self.assertTrue(evaluator.evaluate(predictor) > 0.15)

        for _ in range(1):
            self._trainer.stream_df(tbp)
            evaluator.evaluate(predictor)

        self.assertTrue(evaluator.evaluate(predictor) < 0.05)

    def test_trainer_single_batch_maxq(self):
        new_rl_parameters = ContinuousActionModelParameters(
            rl=self._rl_parameters_maxq,
            training=self._rl_parameters.training,
            knn=self._rl_parameters.knn
        )
        maxq_trainer = ContinuousActionDQNTrainer(
            self._env.normalization, self._env.normalization_action,
            new_rl_parameters
        )

        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            self._env.generate_samples(100000, 1.0)
        predictor = maxq_trainer.predictor()
        tbp = self._env.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )
        evaluator = GridworldContinuousEvaluator(self._env, True)
        self.assertGreater(evaluator.evaluate(predictor), 0.4)

        for _ in range(2):
            maxq_trainer.stream_df(tbp)
            evaluator.evaluate(predictor)

        self.assertLess(evaluator.evaluate(predictor), 0.1)

    def test_evaluator_ground_truth(self):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
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
        self.assertLess(evaluator.mc_loss[-1], 0.1)

    def test_evaluator_timeline(self):
        states, actions, rewards, next_states, next_actions, is_terminal,\
            possible_next_actions, reward_timelines = \
            self._env.generate_samples(100000, 1.0)
        evaluator = Evaluator(self._trainer, DISCOUNT)

        tbp = self._env.preprocess_samples(
            states, actions, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, reward_timelines
        )
        for _ in range(1):
            self._trainer.stream_df(tbp, evaluator)

        self.assertLess(evaluator.td_loss[-1], 0.2)
        self.assertLess(evaluator.mc_loss[-1], 0.2)
