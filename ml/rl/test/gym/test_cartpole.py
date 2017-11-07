from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import random
import numpy as np
import unittest

from ml.rl.training.discrete_action_trainer import \
    DiscreteActionTrainer
from ml.rl.thrift.core.ttypes import RLParameters, \
    TrainingParameters, DiscreteActionModelParameters

from ml.rl.test.gym.run_rl_gym import main
from ml.rl.preprocessing.normalization import \
    NormalizationParameters


def default_normalizer(feats):
    # only for one hot
    normalization = collections.OrderedDict(
        [
            (
                feats[i], NormalizationParameters(
                    feature_type="CONTINUOUS",
                    boxcox_lambda=None,
                    boxcox_shift=0,
                    mean=0,
                    stddev=1
                )
            ) for i in range(len(feats))
        ]
    )
    return normalization


class TestCartpole(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

        actions = ['-1', '+1']
        features = ['angle']
        self._rl_parameters = RLParameters(
            gamma=0.9,
            target_update_rate=1,
            reward_burnin=0,
            maxq_learning=True,
        )
        self._dqn_parameters = DiscreteActionModelParameters(
            rl=self._rl_parameters,
            actions=actions,
            training=TrainingParameters(
                layers=[len(features), 1],
                activations=['linear'],
                minibatch_size=32,
                learning_rate=0.1,
                optimizer='SGD',
            )
        )
        self._trainer = DiscreteActionTrainer(
            default_normalizer(features), self._dqn_parameters
        )

    def test_cartpole(self):
        results = main(
            ['-g', 'CartPole-v0', '-l', '0.05', '-i', '301', '--nosave']
        )
        """
        NOTE: As part of merging this with ml/rl/..., we should guarantee that
        this always gives >195 after training.  Enabling RMSProp (something
        not supported in the open source fork) should do it.
        """
        self.assertTrue(results[-1][1] > 150.0)

    def test_cartpole_dqn_adapted(self):
        results = main(
            [
                '-g', 'CartPole-v0', '-m', 'DQN_ADAPTED', '-l', '0.001', '-i',
                '501', '--nosave', '-o', 'ADAM'
            ]
        )
        self.assertTrue(results[-1][1] > 150.0)

    def test_cartpole_sarsa_adapted(self):
        results = main(
            [
                '-g', 'CartPole-v0', '-m', 'SARSA_ADAPTED', '-l', '0.001', '-i',
                '501', '--nosave', '-o', 'ADAM'
            ]
        )
        self.assertTrue(results[-1][1] > 150.0)
