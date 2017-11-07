from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from ml.rl.test.gridworld.gridworld_base import GridworldBase, S, G


class Gridworld(GridworldBase):
    ACTIONS = ['L', 'R', 'U', 'D']

    grid = np.array(
        [
            [S, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, G],  #
        ]
    )
    width = 5
    height = 5
