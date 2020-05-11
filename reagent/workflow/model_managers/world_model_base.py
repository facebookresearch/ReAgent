#!/usr/bin/env python3

import logging
from typing import Dict, Optional, Tuple

from reagent.core.dataclasses import dataclass
from reagent.gym.policies.policy import Policy
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow.types import Dataset, RewardOptions, RLTrainingOutput, TableSpec


logger = logging.getLogger(__name__)


@dataclass
class WorldModelBase(ModelManager):
    def __post_init_post_parse__(self):
        super().__init__()

    @classmethod
    def normalization_key(cls) -> str:
        raise NotImplementedError()

    def create_policy(self) -> Policy:
        """ Create a WorldModel Policy from env. """
        raise NotImplementedError()

    @property
    def should_generate_eval_dataset(self) -> bool:
        return False

    def _set_normalization_parameters(
        self, normalization_data_map: Dict[str, NormalizationData]
    ):
        """
        Set normalization parameters on current instance
        """
        state_norm_data = normalization_data_map.get(NormalizationKey.STATE, None)
        assert state_norm_data is not None
        assert state_norm_data.dense_normalization_parameters is not None
        action_norm_data = normalization_data_map.get(NormalizationKey.ACTION, None)
        assert action_norm_data is not None
        assert action_norm_data.dense_normalization_parameters is not None
        self.set_normalization_data_map(normalization_data_map)

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
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset], num_epochs: int
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
