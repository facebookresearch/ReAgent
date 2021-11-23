#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

from reagent.workflow.types import Dataset, TableSpec


logger = logging.getLogger(__name__)


class DataFetcher:
    def query_data(
        self,
        input_table_spec: TableSpec,
        discrete_action: bool,
        actions: Optional[List[str]] = None,
        include_possible_actions=True,
        custom_reward_expression: Optional[str] = None,
        sample_range: Optional[Tuple[float, float]] = None,
        multi_steps: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> Dataset:
        raise NotImplementedError()

    def query_data_synthetic_reward(
        self,
        input_table_spec: TableSpec,
        discrete_action_names: Optional[List[str]] = None,
        sample_range: Optional[Tuple[float, float]] = None,
        max_seq_len: Optional[int] = None,
    ) -> Dataset:
        raise NotImplementedError()
