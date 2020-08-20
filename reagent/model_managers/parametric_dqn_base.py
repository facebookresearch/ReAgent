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
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.gym.policies.scorers.discrete_scorer import parametric_dqn_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.models.base import ModelBase
from reagent.parameters import EvaluationParameters, NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.preprocessing.normalization import (
    get_feature_config,
    get_num_output_features,
)
from reagent.preprocessing.types import InputColumn
from reagent.reporting.parametric_dqn_reporter import ParametricDQNReporter
from reagent.training.parametric_dqn_trainer import ParametricDQNTrainer


logger = logging.getLogger(__name__)


@dataclass
class ParametricDQNBase(ModelManager):
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    action_preprocessing_options: Optional[PreprocessingOptions] = None
    state_float_features: Optional[List[Tuple[int, str]]] = None
    action_float_features: Optional[List[Tuple[int, str]]] = None
    reader_options: Optional[ReaderOptions] = None
    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)

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
            self.action_preprocessing_options is None
            or self.action_preprocessing_options.whitelist_features is None
        ), (
            "Please set action whitelist features in action_float_features field of "
            "config instead"
        )

    def create_policy(self, trainer: ParametricDQNTrainer) -> Policy:
        # FIXME: this only works for one-hot encoded actions
        action_dim = trainer.num_gym_actions
        sampler = SoftmaxActionSampler(temperature=self.trainer_param.rl.temperature)
        scorer = parametric_dqn_scorer(
            max_num_actions=action_dim, q_network=trainer.q_network
        )
        return Policy(scorer=scorer, sampler=sampler)

    def create_serving_policy(
        self, normalization_data_map: Dict[str, NormalizationData], trainer
    ) -> Policy:
        # FIXME: this only works for one-hot encoded actions
        action_dim = trainer.num_gym_actions
        return create_predictor_policy_from_model(
            self.build_serving_module(normalization_data_map, trainer),
            max_num_actions=action_dim,
        )

    def get_reporter(self):
        return ParametricDQNReporter()

    @property
    def should_generate_eval_dataset(self) -> bool:
        return False  # Parametric DQN CPE not supported yet

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def action_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.action_float_features)

    def run_feature_identification(
        self, data_fetcher: DataFetcher, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        # Run state feature identification
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

        # Run action feature identification
        action_preprocessing_options = (
            self.action_preprocessing_options or PreprocessingOptions()
        )
        action_features = [
            ffi.feature_id for ffi in self.action_feature_config.float_feature_infos
        ]
        logger.info(f"action whitelist_features: {action_features}")
        action_preprocessing_options = action_preprocessing_options._replace(
            whitelist_features=action_features
        )
        action_normalization_parameters = data_fetcher.identify_normalization_parameters(
            input_table_spec, InputColumn.ACTION, action_preprocessing_options
        )
        return {
            NormalizationKey.STATE: NormalizationData(
                dense_normalization_parameters=state_normalization_parameters
            ),
            NormalizationKey.ACTION: NormalizationData(
                dense_normalization_parameters=action_normalization_parameters
            ),
        }

    @property
    def required_normalization_keys(self) -> List[str]:
        return [NormalizationKey.STATE, NormalizationKey.ACTION]

    def query_data(
        self,
        data_fetcher: DataFetcher,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
    ) -> Dataset:
        raise NotImplementedError()

    def metrics_to_score(self, reward_options: RewardOptions) -> List[str]:
        return get_metrics_to_score(reward_options.metric_reward_values)

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
        raise NotImplementedError()
