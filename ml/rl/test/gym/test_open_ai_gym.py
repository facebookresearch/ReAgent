from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import os

from ml.rl.test.gym.run_gym import main

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestOpenAIGym(unittest.TestCase):

    def _test(self, param_fname, final_score_bar):
        results = main(
            ['-p', "{}/{}".format(BASE_PATH, param_fname), '-f', str(final_score_bar)]
        )
        self.assertGreater(results[-1], final_score_bar)

    def test_maxq_cartpole_v0(self):
        self._test("maxq_cartpole_v0.json", 180.0)
