# Helper script to quickly run RL models on gym via buck.
# Usage: buck run @mode/opt ml/rl:run_gym

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

from ml.rl.test.gym.run_gym import run_gym, USE_CPU

PARAM_JSON = 'ml/rl/test/gym/ddpg_pendulum_v0.json'
SCORE_BAR = -300


if __name__ == '__main__':
    with open(PARAM_JSON, 'r') as f:
        params = json.load(f)
    run_gym(params, SCORE_BAR, USE_CPU)
