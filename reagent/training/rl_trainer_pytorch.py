#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
from typing import Optional

from reagent.core.parameters import RLParameters


logger = logging.getLogger(__name__)


class RLTrainerMixin:
    # Q-value for action that is not possible. Guaranteed to be worse than any
    # legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9

    # todo potential inconsistencies
    _use_seq_num_diff_as_time_diff = None
    _maxq_learning = None
    _multi_steps = None
    # pyre-fixme[13]: Attribute `rl_parameters` is never initialized.
    rl_parameters: RLParameters

    @property
    def gamma(self) -> float:
        return self.rl_parameters.gamma

    @property
    def tau(self) -> float:
        return self.rl_parameters.target_update_rate

    @property
    def multi_steps(self) -> Optional[int]:
        return (
            self.rl_parameters.multi_steps
            if self._multi_steps is None
            else self._multi_steps
        )

    @multi_steps.setter
    def multi_steps(self, multi_steps):
        self._multi_steps = multi_steps

    @property
    def maxq_learning(self) -> bool:
        return (
            self.rl_parameters.maxq_learning
            if self._maxq_learning is None
            else self._maxq_learning
        )

    @maxq_learning.setter
    def maxq_learning(self, maxq_learning):
        self._maxq_learning = maxq_learning

    @property
    def use_seq_num_diff_as_time_diff(self) -> bool:
        return (
            self.rl_parameters.use_seq_num_diff_as_time_diff
            if self._use_seq_num_diff_as_time_diff is None
            else self._use_seq_num_diff_as_time_diff
        )

    @use_seq_num_diff_as_time_diff.setter
    def use_seq_num_diff_as_time_diff(self, use_seq_num_diff_as_time_diff):
        self._use_seq_num_diff_as_time_diff = use_seq_num_diff_as_time_diff

    @property
    def rl_temperature(self) -> float:
        return self.rl_parameters.temperature
