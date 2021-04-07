#!/usr/bin/env python3
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
