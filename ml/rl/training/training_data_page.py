#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class TrainingDataPage(object):
    __slots__ = [
        'state_features', 'action', 'reward', 'next_state_features',
        'next_action', 'possible_next_actions', 'reward_timelines', 'ds'
    ]

    def __init__(
        self, state_features, action, reward, next_state_features, next_action,
        possible_next_actions: np.ndarray, reward_timelines, ds
    ) -> None:
        self.state_features = state_features
        self.action = action
        self.reward = reward
        self.next_state_features = next_state_features
        self.next_action = next_action
        self.possible_next_actions = possible_next_actions
        self.reward_timelines = reward_timelines
        self.ds = ds
