#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import random
from typing import Dict, Optional

from reagent.core.types import RewardOptions
from reagent.data_fetchers.oss_data_fetcher import OssDataFetcher
from reagent.parameters import NormalizationData
from reagent.runners.batch_runner import BatchRunner
from reagent.workflow.model_managers.model_manager import ModelManager


logger = logging.getLogger(__name__)


class OssBatchRunner(BatchRunner):
    def __init__(
        self,
        use_gpu: bool,
        model_manager: ModelManager,
        reward_options: RewardOptions,
        normalization_data_map: Dict[str, NormalizationData],
        warmstart_path: Optional[str] = None,
    ):
        super().__init__(
            use_gpu,
            model_manager,
            OssDataFetcher(),
            reward_options,
            normalization_data_map,
            warmstart_path,
        )
        # Generate a random workflow id for this batch runner
        self.workflow_id = random.randint(1000, 10000000)

    def get_workflow_id(self) -> int:
        return self.workflow_id
