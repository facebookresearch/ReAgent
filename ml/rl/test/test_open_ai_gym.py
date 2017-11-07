from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from ml.rl.test.gym.cartpole_environment import CartpoleV0Environment,\
    CartpoleV1Environment
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.thrift.core.ttypes import\
    RLParameters, TrainingParameters, DiscreteActionModelParameters


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

    def _test_model(
        self,
        env,
        trainer,
        test_run_name,
        final_score_bar,
        num_episodes=301,
        train_and_test_every=10,
        learn_after=10,
        num_learn_batches=100,
        batch_size=1024,
        compute_avg_reward_every=100,
    ):
        avg_over_num_iters = compute_avg_reward_every // train_and_test_every
        reward_history = []
        avg_rewards = []

        predictor = trainer.predictor()

        for i in range(num_episodes):
            env.run_episode(predictor, False)

            if i % train_and_test_every == 0 and i > learn_after:
                for _ in range(num_learn_batches):
                    trainer.stream(*env.get_replay_samples(batch_size))
                evaluation = env.run_episode(predictor, True)
                print("Score on episode {}:".format(i), evaluation)
                reward_history.append(evaluation)
            if i % compute_avg_reward_every == 0:
                avg_recent_rewards = reward_history[-avg_over_num_iters:]
                avg_rewards.append(
                    # Rounding for readability
                    round(np.mean(np.array([avg_recent_rewards])), 2)
                )

        print('Reward history for {}:'.format(test_run_name), reward_history)
        self.assertGreater(avg_recent_rewards[-1], final_score_bar)

    def test_maxq_learning_cartpole_v0(self):
        env = CartpoleV0Environment()
        maxq_discrete_action_parameters = DiscreteActionModelParameters(
            actions=env.actions,
            rl=RLParameters(
                gamma=self.reward_discount_factor,
                target_update_rate=self.target_update_rate,
                reward_burnin=self.reward_burnin,
                maxq_learning=True
            ),
            training=TrainingParameters(
                layers=[-1, 256, 128, 64, -1],
                activations=['relu', 'relu', 'relu', 'linear'],
                minibatch_size=self.minibatch_size,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                gamma=self.lr_decay
            )
        )
        trainer = DiscreteActionTrainer(
            env.normalization, maxq_discrete_action_parameters
        )
        self._test_model(
            env, trainer, "discrete action trainer cartpole v0", 150.0
        )

    def test_maxq_learning_cartpole_v1(self):
        env = CartpoleV1Environment()
        maxq_discrete_action_parameters = DiscreteActionModelParameters(
            actions=env.actions,
            rl=RLParameters(
                gamma=self.reward_discount_factor,
                target_update_rate=self.target_update_rate,
                reward_burnin=self.reward_burnin,
                maxq_learning=True
            ),
            training=TrainingParameters(
                layers=[env.state_dim, 64, 12, 1],
                activations=["relu", "relu", "linear"],
                minibatch_size=self.minibatch_size,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                gamma=self.lr_decay
            )
        )
        trainer = DiscreteActionTrainer(
            env.normalization, maxq_discrete_action_parameters
        )
        self._test_model(
            env, trainer, "discrete action trainer cartpole v1", 400.0
        )
