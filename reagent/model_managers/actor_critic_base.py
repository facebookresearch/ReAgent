#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import reagent.types as rlt
import torch
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
from reagent.evaluation.evaluator import Evaluator, get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.model_managers.model_manager import ModelManager
from reagent.models.base import ModelBase
from reagent.parameters import EvaluationParameters, NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    PolicyNetworkBatchPreprocessor,
    Preprocessor,
)
from reagent.preprocessing.normalization import get_feature_config
from reagent.preprocessing.types import InputColumn
from reagent.reporting.actor_critic_reporter import ActorCriticReporter


logger = logging.getLogger(__name__)


class ActorPolicyWrapper(Policy):
    """ Actor's forward function is our act """

    def __init__(self, actor_network):
        self.actor_network = actor_network

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def act(
        self, obs: rlt.FeatureData, possible_actions_mask: Optional[np.ndarray] = None
    ) -> rlt.ActorOutput:
        self.actor_network.eval()
        output = self.actor_network(obs)
        self.actor_network.train()
        return output.detach().cpu()


@dataclass
class ActorCriticBase(ModelManager):
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    action_preprocessing_options: Optional[PreprocessingOptions] = None
    action_feature_override: Optional[str] = None
    state_float_features: Optional[List[Tuple[int, str]]] = None
    action_float_features: List[Tuple[int, str]] = field(default_factory=list)
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

    @property
    def should_generate_eval_dataset(self) -> bool:
        return False  # CPE not supported in A/C yet

    def create_policy(self, trainer) -> Policy:
        """ Create online actor critic policy. """
        return ActorPolicyWrapper(trainer.actor_network)

    @property
    def metrics_to_score(self, reward_options: RewardOptions) -> List[str]:
        return get_metrics_to_score(reward_options.metric_reward_values)

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def action_feature_config(self) -> rlt.ModelFeatureConfig:
        assert len(self.action_float_features) > 0, "You must set action_float_features"
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

        actor_net_builder = self.actor_net_builder.value
        action_feature_override = actor_net_builder.default_action_preprocessing
        logger.info(f"Default action_feature_override is {action_feature_override}")
        if self.action_feature_override is not None:
            action_feature_override = self.action_feature_override

        assert action_preprocessing_options.feature_overrides is None
        action_preprocessing_options = action_preprocessing_options._replace(
            whitelist_features=action_features,
            feature_overrides={fid: action_feature_override for fid in action_features},
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
        logger.info("Starting query")
        return data_fetcher.query_data(
            input_table_spec=input_table_spec,
            discrete_action=False,
            include_possible_actions=False,
            custom_reward_expression=reward_options.custom_reward_expression,
            sample_range=sample_range,
        )

    def get_reporter(self):
        return ActorCriticReporter()

    def build_batch_preprocessor(
        self,
        reader_options: ReaderOptions,
        use_gpu: bool,
        batch_size: int,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> BatchPreprocessor:
        state_preprocessor = Preprocessor(
            normalization_data_map[
                NormalizationKey.STATE
            ].dense_normalization_parameters,
            use_gpu=use_gpu,
        )
        action_preprocessor = Preprocessor(
            normalization_data_map[
                NormalizationKey.ACTION
            ].dense_normalization_parameters,
            use_gpu=use_gpu,
        )
        return PolicyNetworkBatchPreprocessor(
            state_preprocessor=state_preprocessor,
            action_preprocessor=action_preprocessor,
            use_gpu=use_gpu,
        )
