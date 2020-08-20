#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

from reagent.core.dataclasses import dataclass
from reagent.core.rl_training_output import RLTrainingOutput
from reagent.core.types import Dataset, ReaderOptions, RewardOptions, TableSpec
from reagent.data_fetchers.data_fetcher import DataFetcher
from reagent.model_managers.model_manager import ModelManager
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.reporting.world_model_reporter import WorldModelReporter


logger = logging.getLogger(__name__)


@dataclass
class WorldModelBase(ModelManager):
    def __post_init_post_parse__(self):
        super().__init__()

    @classmethod
    def normalization_key(cls) -> str:
        raise NotImplementedError()

    @property
    def should_generate_eval_dataset(self) -> bool:
        return False

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE, NormalizationKey.ACTION]

    def run_feature_identification(
        self, data_fetcher: DataFetcher, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        raise NotImplementedError()

    def get_reporter(self):
        return WorldModelReporter()

    def query_data(
        self,
        data_fetcher: DataFetcher,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        raise NotImplementedError()

    def build_batch_preprocessor(
        self,
        reader_options: ReaderOptions,
        use_gpu: bool,
        batch_size: int,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> BatchPreprocessor:
        raise NotImplementedError()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        num_epochs: int,
        reader_options: ReaderOptions,
    ) -> RLTrainingOutput:
        """
        Train the model

        Returns partially filled RLTrainingOutput. The field that should not be filled
        are:
        - output_path
        - warmstart_output_path
        - vis_metrics
        - validation_output
        """
        raise NotImplementedError()
