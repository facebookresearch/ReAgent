from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from ml.rl.test.gym.run_gym import run_gym, USE_CPU


class TestOpenAIGym(unittest.TestCase):
    def _test(self, parameters_string, score_bar):
        results = run_gym(parameters_string, score_bar, USE_CPU)
        self.assertGreater(results[-1], score_bar)

    def _get_discrete_base_test_config(self):
        base_config_dict = {
            'env': 'CartPole-v0',
            'model_type': 'discrete',
            'rl': {
                'reward_discount_factor': 0.99,
                'target_update_rate': 0.2,
                'reward_burnin': 1,
                'maxq_learning': 1,
                'epsilon': 0.0,
                'temperature': .35
            },
            'training': {
                'layers': [
                    -1,
                    128,
                    64,
                    -1
                ],
                'activations': [
                    'relu',
                    'relu',
                    'linear'
                ],
                'minibatch_size': 128,
                'learning_rate': 0.01,
                'optimizer': 'ADAM',
                'learning_rate_decay': 0.999
            },
            'run_details': {
                'num_episodes': 1001,
                'train_every': 10,
                'train_after': 10,
                'test_every': 100,
                'test_after': 10,
                'num_train_batches': 15,
                'avg_over_num_episodes': 100,
                'render': 0,
                'render_every': 100
            }
        }
        return base_config_dict

    def test_discrete_qlearning_softmax_cartpole_v0(self):
        """Test discrete action q-learning model on cartpole using
        a softmax policy."""
        config = self._get_discrete_base_test_config()
        config['model_type'] = 'discrete'
        config['rl']['maxq_learning'] = 1
        self._test(config, 180)

    def test_discrete_sarsa_softmax_cartpole_v0(self):
        """Test discrete action sarsa model on cartpole using
        a softmax policy."""
        config = self._get_discrete_base_test_config()
        config['model_type'] = 'discrete'
        config['rl']['maxq_learning'] = 0
        config['training']['minibatch_size'] = 200
        self._test(config, 180)

    def test_parametric_qlearning_softmax_cartpole_v0(self):
        """Test parametric action q-learning model on cartpole using
        a softmax policy."""
        config = self._get_discrete_base_test_config()
        config['model_type'] = 'parametric'
        config['rl']['maxq_learning'] = 1
        self._test(config, 180)

    def test_parametric_sarsa_softmax_cartpole_v0(self):
        """Test parametric action sarsa model on cartpole using
        a softmax policy."""
        config = self._get_discrete_base_test_config()
        config['model_type'] = 'parametric'
        config['rl']['maxq_learning'] = 0
        self._test(config, 180)

    def test_ddpg_pendulum_v0(self):
        """Test deep deterministic policy gradient algo on Pendulum-v0."""
        config = {
            'env': 'Pendulum-v0',
            'model_type': 'continuous',
            'rl': {
                'reward_discount_factor': .99,
                'epsilon': 0.0,
            },
            'actor_training': {
                'layers': [
                    -1,
                    128,
                    64,
                    -1
                ],
                'activations': [
                    'relu',
                    'relu',
                    'tanh'
                ],
                'minibatch_size': 128,
                'learning_rate': 0.01,
                'optimizer': 'ADAM',
                'learning_rate_decay': 0.999
            },
            'critic_training': {
                'learning_rate_decay': 0.999
            },
            'run_details': {
                'num_episodes': 300,
            }
        }
        self._test(config, -1300)
