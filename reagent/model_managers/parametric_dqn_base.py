#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

import reagent.core.types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
)
from reagent.data.data_fetcher import DataFetcher
from reagent.data.reagent_data_module import ReAgentDataModule
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.gym.policies.scorers.discrete_scorer import parametric_dqn_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.models.base import ModelBase
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.preprocessing.normalization import (
    get_feature_config,
    get_num_output_features,
)
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
        super().__post_init_post_parse__()
        assert (
            self.state_preprocessing_options is None
            or self.state_preprocessing_options.allowedlist_features is None
        ), (
            "Please set state whitelist features in state_float_features field of "
            "config instead"
        )
        assert (
            self.action_preprocessing_options is None
            or self.action_preprocessing_options.allowedlist_features is None
        ), (
            "Please set action whitelist features in action_float_features field of "
            "config instead"
        )
        self._state_preprocessing_options = self.state_preprocessing_options
        self._action_preprocessing_options = self.action_preprocessing_options
        self._q_network: Optional[ModelBase] = None
        self._metrics_to_score: Optional[List[str]] = None

    def create_policy(self, serving: bool) -> Policy:
        """ Create an online DiscreteDQN Policy from env. """

        # FIXME: this only works for one-hot encoded actions
        action_dim = get_num_output_features(
            self.action_normalization_data.dense_normalization_parameters
        )
        if serving:
            return create_predictor_policy_from_model(
                self.build_serving_module(), max_num_actions=action_dim
            )
        else:
            sampler = SoftmaxActionSampler(temperature=self.rl_parameters.temperature)
            scorer = parametric_dqn_scorer(
                max_num_actions=action_dim, q_network=self._q_network
            )
            return Policy(scorer=scorer, sampler=sampler)

    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.eval_parameters.calc_cpe_in_training

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def action_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.action_float_features)

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        # Run state feature identification
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

        # Run action feature identification
        action_preprocessing_options = (
            self._action_preprocessing_options or PreprocessingOptions()
        )
        action_features = [
            ffi.feature_id for ffi in self.action_feature_config.float_feature_infos
        ]
        logger.info(f"action allowedlist_features: {action_features}")
        action_preprocessing_options = action_preprocessing_options._replace(
            allowedlist_features=action_features
        )
        action_normalization_parameters = identify_normalization_parameters(
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
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
    ) -> Dataset:
        raise NotImplementedError()

    @property
    def metrics_to_score(self) -> List[str]:
        assert self.reward_options is not None
        if self._metrics_to_score is None:
            # pyre-fixme[16]: `ParametricDQNBase` has no attribute `_metrics_to_score`.
            self._metrics_to_score = get_metrics_to_score(
                self._reward_options.metric_reward_values
            )
        return self._metrics_to_score

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
        raise NotImplementedError()
