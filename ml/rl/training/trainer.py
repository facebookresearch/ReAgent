#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List

import ml.rl.types as rlt


logger = logging.getLogger(__name__)


class Trainer:
    def train(self, training_batch: rlt.PreprocessedTrainingBatch) -> None:
        raise NotImplementedError()

    def state_dict(self):
        return {c: getattr(self, c).state_dict() for c in self.warm_start_components()}

    def load_state_dict(self, state_dict):
        for c in self.warm_start_components():
            getattr(self, c).load_state_dict(state_dict[c])

    def warm_start_components(self) -> List[str]:
        """
        The trainer should specify what members to save and load
        """
        raise NotImplementedError
