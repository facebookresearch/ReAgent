from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from ml.rl.test.gym.open_ai_gym_environment import OpenAIGymEnvironment
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.thrift.core.ttypes import\
    RLParameters, TrainingParameters, DiscreteActionModelParameters
from ml.rl.test.gym.run_gym import run


class TestOpenAIGym(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()

        self.lr_decay = 0.999
        self.learning_rate = 0.005
        self.target_update_rate = 0.1
        self.optimizer = "ADAM"
        self.reward_discount_factor = 0.99
        self.minibatch_size = 128
        self.reward_burnin = 10

    def test_maxq_learning_cartpole_v0(self):
        env = OpenAIGymEnvironment('CartPole-v0')
        maxq_discrete_action_parameters = DiscreteActionModelParameters(
            actions=env.actions,
            rl=RLParameters(
                gamma=self.reward_discount_factor,
                target_update_rate=self.target_update_rate,
                reward_burnin=self.reward_burnin,
                maxq_learning=True
            ),
            training=TrainingParameters(
                layers=[-1, 128, 64, -1],
                activations=['relu', 'relu', 'linear'],
                minibatch_size=self.minibatch_size,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                gamma=self.lr_decay
            )
        )
        trainer = DiscreteActionTrainer(
            env.normalization, maxq_discrete_action_parameters
        )
        final_score_bar = 150.0
        avg_recent_rewards = run(
            env, trainer, "discrete action trainer cartpole v0",
            final_score_bar
        )
        self.assertGreater(avg_recent_rewards[-1], final_score_bar)

