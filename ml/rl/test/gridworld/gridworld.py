#!/usr/bin/env python3


from typing import Dict, List

from ml.rl.test.gridworld.gridworld_base import GridworldBase, Samples
from ml.rl.training.training_data_page import TrainingDataPage


class Gridworld(GridworldBase):
    def generate_samples(self, num_transitions, epsilon, discount_factor) -> Samples:
        samples = self.generate_samples_discrete(
            num_transitions, epsilon, discount_factor
        )
        return samples

    def preprocess_samples(
        self, samples: Samples, minibatch_size: int, one_hot_action: bool = True
    ) -> List[TrainingDataPage]:
        return self.preprocess_samples_discrete(samples, minibatch_size, one_hot_action)
