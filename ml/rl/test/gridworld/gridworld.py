#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import List, Optional, Union

from ml.rl.test.environment.environment import MultiStepSamples
from ml.rl.test.gridworld.gridworld_base import GridworldBase, Samples
from ml.rl.training.training_data_page import TrainingDataPage


class Gridworld(GridworldBase):
    def state_to_features(self, state):
        return {state: 1.0}

    def generate_samples(
        self,
        num_transitions,
        epsilon,
        discount_factor,
        multi_steps: Optional[int] = None,
    ) -> Union[Samples, MultiStepSamples]:
        return self.generate_random_samples(
            num_transitions,
            use_continuous_action=False,
            epsilon=epsilon,
            multi_steps=multi_steps,
        )

    def preprocess_samples(
        self,
        samples: Samples,
        minibatch_size: int,
        one_hot_action: bool = True,
        use_gpu: bool = False,
        do_shuffle: bool = True,
    ) -> List[TrainingDataPage]:
        return self.preprocess_samples_discrete(
            samples,
            minibatch_size,
            one_hot_action,
            use_gpu=use_gpu,
            do_shuffle=do_shuffle,
        )
