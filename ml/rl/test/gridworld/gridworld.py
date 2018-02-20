#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Tuple, Dict, List, Optional

from ml.rl.test.gridworld.gridworld_base import GridworldBase
from ml.rl.training.training_data_page import \
    TrainingDataPage


class Gridworld(GridworldBase):
    def generate_samples(
        self, num_transitions, epsilon, with_possible=True
    ) -> Tuple[List[Dict[int, float]], List[str], List[float], List[
        Dict[int, float]
    ], List[str], List[bool], List[List[str]], List[Dict[int, float]]]:
        return self.generate_samples_discrete(
            num_transitions, epsilon, with_possible
        )

    def preprocess_samples(
        self,
        states: List[Dict[int, float]],
        actions: List[str],
        rewards: List[float],
        next_states: List[Dict[int, float]],
        next_actions: List[str],
        is_terminals: List[bool],
        possible_next_actions: List[List[str]],
        reward_timelines: Optional[List[Dict[int, float]]],
        minibatch_size: int,
    ) -> List[TrainingDataPage]:
        return self.preprocess_samples_discrete(
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            is_terminals,
            possible_next_actions,
            reward_timelines,
            minibatch_size,
        )
