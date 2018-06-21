#!/usr/bin/env python3


from typing import Dict, List

from ml.rl.test.gridworld.gridworld_base import GridworldBase, Samples
from ml.rl.training.training_data_page import TrainingDataPage


class Gridworld(GridworldBase):
    def generate_samples(self, num_transitions, epsilon, with_possible=True) -> Samples:
        samples = self.generate_samples_discrete(
            num_transitions, epsilon, with_possible
        )
        return samples

    def preprocess_samples(
        self, samples: Samples, minibatch_size: int
    ) -> List[TrainingDataPage]:
        return self.preprocess_samples_discrete(samples, minibatch_size)
