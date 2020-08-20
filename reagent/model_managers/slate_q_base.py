#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

import reagent.types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)
from reagent.data_fetchers.data_fetcher import DataFetcher
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.top_k_sampler import TopKSampler
from reagent.gym.policies.scorers.slate_q_scorer import slate_q_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.preprocessing.normalization import get_feature_config
from reagent.preprocessing.types import InputColumn
from reagent.reporting.ranking_model_reporter import RankingModelReporter
from reagent.training import SlateQTrainerParameters


logger = logging.getLogger(__name__)


@dataclass
class SlateQBase(ModelManager):
    slate_feature_id: int = -1
    slate_score_id: Tuple[int, int] = (-1, -1)
    item_preprocessing_options: Optional[PreprocessingOptions] = None
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    state_float_features: Optional[List[Tuple[int, str]]] = None
    item_float_features: Optional[List[Tuple[int, str]]] = None
    slate_size: int = -1
    num_candidates: int = -1
    trainer_param: SlateQTrainerParameters = field(
        default_factory=SlateQTrainerParameters
    )

    def __post_init_post_parse__(self):
        super().__init__()
        assert (
            self.state_preprocessing_options is None
            or self.state_preprocessing_options.whitelist_features is None
        ), (
            "Please set state whitelist features in state_float_features field of "
            "config instead"
        )
        assert (
            self.item_preprocessing_options is None
            or self.item_preprocessing_options.whitelist_features is None
        ), (
            "Please set item whitelist features in item_float_features field of "
            "config instead"
        )
        assert (
            self.item_preprocessing_options is None
            or self.item_preprocessing_options.sequence_feature_id is None
        ), "Please set slate_feature_id field of config instead"
        self.eval_parameters = self.trainer_param.evaluation

    def create_policy(self, trainer) -> Policy:
        scorer = slate_q_scorer(
            num_candidates=self.num_candidates, q_network=trainer.q_network
        )
        sampler = TopKSampler(k=self.slate_size)
        return Policy(scorer=scorer, sampler=sampler)

    def create_serving_policy(
        self, normalization_data_map: Dict[str, NormalizationData], trainer
    ) -> Policy:
        return create_predictor_policy_from_model(
            self.build_serving_module(normalization_data_map, trainer),
            max_num_actions=self.num_candidates,
            slate_size=self.slate_size,
        )

    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.eval_parameters.calc_cpe_in_training

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def item_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.item_float_features)

    def get_reporter(self):
        return RankingModelReporter()

    def run_feature_identification(
        self, data_fetcher: DataFetcher, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        state_preprocessing_options = (
            self.state_preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id for ffi in self.state_feature_config.float_feature_infos
        ]
        logger.info(f"state whitelist_features: {state_features}")
        state_preprocessing_options = state_preprocessing_options._replace(
            whitelist_features=state_features
        )
        state_normalization_parameters = data_fetcher.identify_normalization_parameters(
            input_table_spec, InputColumn.STATE_FEATURES, state_preprocessing_options
        )
        item_preprocessing_options = (
            self.item_preprocessing_options or PreprocessingOptions()
        )
        item_features = [
            ffi.feature_id for ffi in self.item_feature_config.float_feature_infos
        ]
        logger.info(f"item whitelist_features: {item_features}")
        item_preprocessing_options = item_preprocessing_options._replace(
            whitelist_features=item_features, sequence_feature_id=self.slate_feature_id
        )
        item_normalization_parameters = data_fetcher.identify_normalization_parameters(
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

    def build_batch_preprocessor(
        self,
        reader_options: ReaderOptions,
        use_gpu: bool,
        batch_size: int,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> BatchPreprocessor:
        raise NotImplementedError("Write for OSS")

    def query_data(
        self,
        data_fetcher: DataFetcher,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        raise NotImplementedError("Write for OSS")

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        num_epochs: int,
        reader_options: ReaderOptions,
    ) -> RLTrainingOutput:
        raise NotImplementedError("Write for OSS")
