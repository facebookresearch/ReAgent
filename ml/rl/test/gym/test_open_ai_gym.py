from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import unittest

from ml.rl.test.gym.run_gym import run_gym, USE_CPU


class TestOpenAIGym(unittest.TestCase):
    def _test(self, parameters_string, score_bar):
        results = run_gym(json.loads(parameters_string), score_bar, USE_CPU)
        self.assertGreater(results[-1], score_bar)

    def test_discrete_softmax_cartpole_v0(self):
        """Test discrete action deep q-learning model on cartpole using
        a softmax policy."""
        self._test(
            """
            {
              "env": "CartPole-v0",
              "model_type": "discrete",
              "rl": {
                "reward_discount_factor": 0.99,
                "target_update_rate": 0.2,
                "reward_burnin": 1,
                "maxq_learning": 1,
                "epsilon": 0.0,
                "temperature": 1
              },
              "training": {
                "layers": [
                  -1,
                  128,
                  64,
                  -1
                ],
                "activations": [
                  "relu",
                  "relu",
                  "linear"
                ],
                "minibatch_size": 256,
                "learning_rate": 0.01,
                "optimizer": "ADAM",
                "learning_rate_decay": 0.999
              },
              "run_details": {
                "num_episodes": 701,
                "train_every": 10,
                "train_after": 10,
                "test_every": 100,
                "test_after": 10,
                "num_train_batches": 10,
                "avg_over_num_episodes": 100,
                "render": 0,
                "render_every": 100
              }
            }
            """, 180.0
        )

    def test_parametric_softmax_cartpole_v0(self):
        """Test parametric action deep q-learning model on cartpole using
        a softmax policy."""
        self._test(
            """
            {
              "env": "CartPole-v0",
              "model_type": "parametric",
              "rl": {
                "reward_discount_factor": 0.99,
                "target_update_rate": 0.2,
                "reward_burnin": 1,
                "maxq_learning": 1,
                "epsilon": 0.0,
                "temperature": 1
              },
              "training": {
                "layers": [
                  -1,
                  128,
                  64,
                  -1
                ],
                "activations": [
                  "relu",
                  "relu",
                  "linear"
                ],
                "minibatch_size": 256,
                "learning_rate": 0.01,
                "optimizer": "ADAM",
                "learning_rate_decay": 0.999
              },
              "run_details": {
                "num_episodes": 701,
                "train_every": 10,
                "train_after": 10,
                "test_every": 100,
                "test_after": 10,
                "num_train_batches": 10,
                "avg_over_num_episodes": 100,
                "render": 0,
                "render_every": 100
              }
            }
            """, 180.0
        )

    # def test_asteroids_v0(self):
    #     self._test("maxq_asteroids_v0.json", 200.0)
