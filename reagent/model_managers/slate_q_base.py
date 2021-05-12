#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Tuple

import reagent.core.types as rlt
from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, NormalizationKey
from reagent.data.data_fetcher import DataFetcher
from reagent.data.reagent_data_module import ReAgentDataModule
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.top_k_sampler import TopKSampler
from reagent.gym.policies.scorers.slate_q_scorer import slate_q_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.models.base import ModelBase
from reagent.preprocessing.normalization import get_feature_config
from reagent.preprocessing.types import InputColumn
from reagent.reporting.slate_q_reporter import SlateQReporter
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


logger = logging.getLogger(__name__)


@dataclass
class SlateQBase(ModelManager):
    slate_feature_id: int = 0
    slate_score_id: Tuple[int, int] = (0, 0)
    item_preprocessing_options: Optional[PreprocessingOptions] = None
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    state_float_features: Optional[List[Tuple[int, str]]] = None
    item_float_features: Optional[List[Tuple[int, str]]] = None

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        assert (
            self.state_preprocessing_options is None
            or self.state_preprocessing_options.allowedlist_features is None
        ), (
            "Please set state whitelist features in state_float_features field of "
            "config instead"
        )
        assert (
            self.item_preprocessing_options is None
            or self.item_preprocessing_options.allowedlist_features is None
        ), (
            "Please set item whitelist features in item_float_features field of "
            "config instead"
        )
        assert (
            self.item_preprocessing_options is None
            or self.item_preprocessing_options.sequence_feature_id is None
        ), "Please set slate_feature_id field of config instead"
        self._state_preprocessing_options = self.state_preprocessing_options
        self._item_preprocessing_options = self.item_preprocessing_options
        self._q_network: Optional[ModelBase] = None
        self.eval_parameters = self.trainer_param.evaluation

    def create_policy(self, serving: bool) -> Policy:
        if serving:
            return create_predictor_policy_from_model(
                self.build_serving_module(),
                max_num_actions=self.num_candidates,
                slate_size=self.slate_size,
            )
        else:
            scorer = slate_q_scorer(
                num_candidates=self.num_candidates, q_network=self._q_network
            )
            sampler = TopKSampler(k=self.slate_size)
            return Policy(scorer=scorer, sampler=sampler)

    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.eval_parameters.calc_cpe_in_training

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def item_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.item_float_features)

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        state_preprocessing_options = (
            self._state_preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id for ffi in self.state_feature_config.float_feature_infos
        ]
        logger.info(f"state allowedlist_features: {state_features}")
        state_preprocessing_options = state_preprocessing_options._replace(
            allowedlist_features=state_features
        )
        state_normalization_parameters = identify_normalization_parameters(
            input_table_spec, InputColumn.STATE_FEATURES, state_preprocessing_options
        )
        item_preprocessing_options = (
            self._item_preprocessing_options or PreprocessingOptions()
        )
        item_features = [
            ffi.feature_id for ffi in self.item_feature_config.float_feature_infos
        ]
        logger.info(f"item allowedlist_features: {item_features}")
        item_preprocessing_options = item_preprocessing_options._replace(
            allowedlist_features=item_features,
            sequence_feature_id=self.slate_feature_id,
        )
        item_normalization_parameters = identify_normalization_parameters(
            input_table_spec,
            InputColumn.STATE_SEQUENCE_FEATURES,
            item_preprocessing_options,
        )
        return {
            NormalizationKey.STATE: NormalizationData(
                dense_normalization_parameters=state_normalization_parameters
            ),
            NormalizationKey.ITEM: NormalizationData(
                dense_normalization_parameters=item_normalization_parameters
            ),
        }

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE, NormalizationKey.ITEM]

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
    ) -> Dataset:
        raise NotImplementedError("Write for OSS")

    def get_reporter(self):
        return SlateQReporter()

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
        raise NotImplementedError("Write for OSS")
