#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Dict, List, Optional
import numpy as np

from ml.rl.preprocessing.normalization import NormalizationParameters

from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.training.training_data_page import TrainingDataPage


class GridworldEnum(Gridworld):
    @property
    def normalization(self):
        return {
            'f':
            NormalizationParameters(
                feature_type="ENUM",
                boxcox_lambda=None,
                boxcox_shift=None,
                mean=None,
                stddev=None,
                possible_values=[float(i) for i in range(len(self.STATES))],
                quantiles=None,
            )
        }

    def preprocess_samples(
        self,
        states: List[Dict[str, float]],
        actions: List[str],
        rewards: List[float],
        next_states: List[Dict[str, float]],
        next_actions: List[str],
        is_terminals: List[bool],
        possible_next_actions: List[List[str]],
        reward_timelines: Optional[List[Dict[int, float]]],
    ) -> TrainingDataPage:
        tdp = self.preprocess_samples_discrete(
            states, actions, rewards, next_states, next_actions, is_terminals,
            possible_next_actions, reward_timelines
        )
        tdp.states = np.where(tdp.states == 1.0)[1].reshape(-1, 1
                                                           ).astype(np.float32)
        tdp.next_states = np.where(tdp.next_states == 1.0)[1].reshape(
            -1, 1
        ).astype(np.float32)
        return tdp
