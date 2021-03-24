#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Tuple

from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, NormalizationKey
from reagent.gym.policies.policy import Policy
from reagent.model_managers.model_manager import ModelManager
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.workflow.data import ReAgentDataModule
from reagent.workflow.types import (
    Dataset,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)


logger = logging.getLogger(__name__)


@dataclass
class WorldModelBase(ModelManager):
    reward_boost: Optional[Dict[str, float]] = None

    @classmethod
    def normalization_key(cls) -> str:
        raise NotImplementedError()

    def create_policy(self) -> Policy:
        """ Create a WorldModel Policy from env. """
        raise NotImplementedError()

    @property
    def should_generate_eval_dataset(self) -> bool:
        return False

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE, NormalizationKey.ACTION]

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        raise NotImplementedError()

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        raise NotImplementedError()

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        raise NotImplementedError()

    def train(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: Optional[ResourceOptions] = None,
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
