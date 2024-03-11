#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging
from dataclasses import replace
from typing import Dict, Optional, Tuple

from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, NormalizationKey
from reagent.data.data_fetcher import DataFetcher
from reagent.data.manual_data_module import ManualDataModule
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.preprocessing.types import InputColumn
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    RewardOptions,
    TableSpec,
)

try:
    from reagent.model_managers.fb.model_manager import ModelManager
except ImportError:
    from reagent.model_managers.model_manager import ModelManager


logger = logging.getLogger(__name__)


@dataclass
class WorldModelBase(ModelManager):
    reward_boost: Optional[Dict[str, float]] = None


class WorldModelDataModule(ManualDataModule):
    @property
    def should_generate_eval_dataset(self) -> bool:
        return False

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        # Run state feature identification
        state_preprocessing_options = PreprocessingOptions()
        state_features = [
            ffi.feature_id
            for ffi in self.model_manager.state_feature_config.float_feature_infos
        ]
        logger.info(f"Overriding state allowedlist_features: {state_features}")
        assert len(state_features) > 0, "No state feature is specified"
        state_preprocessing_options = replace(
            state_preprocessing_options, allowedlist_features=state_features
        )

        state_normalization_parameters = identify_normalization_parameters(
            input_table_spec, InputColumn.STATE_FEATURES, state_preprocessing_options
        )

        return {
            NormalizationKey.STATE: NormalizationData(
                dense_normalization_parameters=state_normalization_parameters
            )
        }

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
    ) -> Dataset:
        raise NotImplementedError()

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        raise NotImplementedError()
