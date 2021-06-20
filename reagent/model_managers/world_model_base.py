#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Tuple

from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, NormalizationKey
from reagent.data.data_fetcher import DataFetcher
from reagent.data.manual_data_module import ManualDataModule
from reagent.data.reagent_data_module import ReAgentDataModule
from reagent.gym.policies.policy import Policy
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.preprocessing.types import InputColumn
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
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

    # TODO: Add get_data_module() method once methods in
    # `WorldModelDataModule` class are implemented
    # def get_data_module(
    #     self,
    #     *,
    #     input_table_spec: Optional[TableSpec] = None,
    #     reward_options: Optional[RewardOptions] = None,
    #     reader_options: Optional[ReaderOptions] = None,
    #     setup_data: Optional[Dict[str, bytes]] = None,
    #     saved_setup_data: Optional[Dict[str, bytes]] = None,
    #     resource_options: Optional[ResourceOptions] = None,
    # ) -> Optional[ReAgentDataModule]:
    #     return WorldModelDataModule(
    #         input_table_spec=input_table_spec,
    #         reward_options=reward_options,
    #         setup_data=setup_data,
    #         saved_setup_data=saved_setup_data,
    #         reader_options=reader_options,
    #         resource_options=resource_options,
    #         model_manager=self,
    #     )

    def train(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: ResourceOptions,
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
        logger.info(f"state allowedlist_features: {state_features}")
        state_preprocessing_options = state_preprocessing_options._replace(
            allowedlist_features=state_features
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
