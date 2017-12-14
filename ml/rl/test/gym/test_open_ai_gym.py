from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import os

from libfb import parutil  # type: ignore

from ml.rl.test.gym.run_gym import main

PARAM_ROOT = 'ml/rl/test/gym'


def get_file_path(file_name):
    assert file_name is not None, 'A file is required'
    path = os.path.join(PARAM_ROOT, file_name)
    return parutil.get_file_path(path)


class TestOpenAIGym(unittest.TestCase):

    def _test(self, param_fname, score_bar):
        param_file_path = get_file_path(param_fname)
        results = main(
            ['-p', param_file_path, '-s', str(score_bar)]
        )
        self.assertGreater(results[-1], score_bar)

    def test_maxq_cartpole_v0(self):
        self._test("maxq_cartpole_v0.json", 180.0)
