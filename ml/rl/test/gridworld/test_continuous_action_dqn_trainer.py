from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# @build:deps [
# @/caffe2/caffe2/python:caffe2_py
# @/caffe2/proto:fb_protobuf
# @/hiveio:par_init
# ]

import random
import numpy as np
import unittest

from ml.rl.thrift.core.ttypes import \
    RLParameters, TrainingParameters, \
    ContinuousActionModelParameters, KnnParameters
from ml.rl.training.continuous_action_dqn_trainer import \
    ContinuousActionDQNTrainer
from ml.rl.test.utils import default_normalizer


class MockEnv:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

    @property
    def normalization(self):
        return default_normalizer([i for i in range(self.num_states)])

    @property
    def normalization_action(self):
        return default_normalizer([i for i in range(self.num_actions)])


class TestGridworldContinuous(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

        self.state_dim, self.action_dim = 2, 3

        self._env = MockEnv(self.state_dim, self.action_dim)

        self._rl_parameters = RLParameters(
            gamma=0.9,
            target_update_rate=0.5,
            reward_burnin=10,
            maxq_learning=False,
        )
        self._rl_parameters_maxq = RLParameters(
            gamma=0.9,
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
            )
        )
        self._trainer = ContinuousActionDQNTrainer(
            self._env.normalization, self._env.normalization_action,
            self._rl_parameters
        )

    def test_get_max_q_labels(self):
        batch_size = 10
        next_states = np.array(
            [self.generate_state(i) for i in range(batch_size)], dtype=np.float32
        )
        possible_next_actions = [self.generate_pna(i) for i in range(batch_size)]

        max_q_values = self._trainer.get_max_q_values(
            next_states, possible_next_actions
        )

        for i in range(batch_size):
            next_state = next_states[i]
            pnas = possible_next_actions[i]

            expected_max_q_value = None
            if pnas.shape[0] > 0:
                for pna in pnas:
                    # We haven't enabled slow target updates so using our score
                    # model should yield the same results as our target network
                    q_value = self._trainer.get_q_values(
                        [next_state], [pna]
                    )[0][0]
                    tn_q_value = self._trainer.target_network.target_values(
                        np.concatenate([next_state, pna]).reshape((1, -1))
                    )[0][0]
                    self.assertAlmostEquals(q_value, tn_q_value, places=3)
                    expected_max_q_value = (
                        q_value if expected_max_q_value is None
                        else max(q_value, expected_max_q_value)
                    )
            else:
                expected_max_q_value = 0
            self.assertAlmostEquals(
                max_q_values[i][0],
                expected_max_q_value,
                places=3
            )

    def generate_state(self, i):
        return [i] * self.state_dim

    def generate_pna(self, i):
        num_pna = (i % 3)
        rtn = np.ones((num_pna, self.action_dim), dtype=np.float32) * 0.1 * i
        for i in range(len(rtn)):
            rtn[i] += i * 0.01
        return rtn
