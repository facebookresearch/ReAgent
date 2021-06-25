#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import reagent.core.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
)
from reagent.data import DataFetcher, ReAgentDataModule, ManualDataModule
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.model_managers.model_manager import ModelManager
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    PolicyNetworkBatchPreprocessor,
    Preprocessor,
)
from reagent.preprocessing.normalization import get_feature_config
from reagent.preprocessing.types import InputColumn
from reagent.reporting.actor_critic_reporter import ActorCriticReporter
from reagent.training import ReAgentLightningModule
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


class ActorPolicyWrapper(Policy):
    """Actor's forward function is our act"""

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
    save_critic_bool: bool = True

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

    def create_policy(
        self,
        trainer_module: ReAgentLightningModule,
        serving: bool = False,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    ) -> Policy:
        """Create online actor critic policy."""

        if serving:
            assert normalization_data_map
            return create_predictor_policy_from_model(
                self.build_serving_module(trainer_module, normalization_data_map)
            )
        else:
            return ActorPolicyWrapper(trainer_module.actor_network)

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def action_feature_config(self) -> rlt.ModelFeatureConfig:
        assert len(self.action_float_features) > 0, "You must set action_float_features"
        return get_feature_config(self.action_float_features)

    def get_state_preprocessing_options(self) -> PreprocessingOptions:
        state_preprocessing_options = (
            self.state_preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id for ffi in self.state_feature_config.float_feature_infos
        ]
        logger.info(f"state allowedlist_features: {state_features}")
        state_preprocessing_options = state_preprocessing_options._replace(
            allowedlist_features=state_features
        )
        return state_preprocessing_options

    def get_action_preprocessing_options(self) -> PreprocessingOptions:
        action_preprocessing_options = (
            self.action_preprocessing_options or PreprocessingOptions()
        )
        action_features = [
            ffi.feature_id for ffi in self.action_feature_config.float_feature_infos
        ]
        logger.info(f"action allowedlist_features: {action_features}")

        # pyre-fixme
        actor_net_builder = self.actor_net_builder.value
        action_feature_override = actor_net_builder.default_action_preprocessing
        logger.info(f"Default action_feature_override is {action_feature_override}")
        if self.action_feature_override is not None:
            action_feature_override = self.action_feature_override

        assert action_preprocessing_options.feature_overrides is None
        action_preprocessing_options = action_preprocessing_options._replace(
            allowedlist_features=action_features,
            feature_overrides={fid: action_feature_override for fid in action_features},
        )
        return action_preprocessing_options

    def get_data_module(
        self,
        *,
        input_table_spec: Optional[TableSpec] = None,
        reward_options: Optional[RewardOptions] = None,
        reader_options: Optional[ReaderOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        resource_options: Optional[ResourceOptions] = None,
    ) -> Optional[ReAgentDataModule]:
        return ActorCriticDataModule(
            input_table_spec=input_table_spec,
            reward_options=reward_options,
            setup_data=setup_data,
            saved_setup_data=saved_setup_data,
            reader_options=reader_options,
            resource_options=resource_options,
            model_manager=self,
        )

    def get_reporter(self):
        return ActorCriticReporter()


class ActorCriticDataModule(ManualDataModule):
    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        """
        Derive preprocessing parameters from data.
        """
        # Run state feature identification
        state_normalization_parameters = identify_normalization_parameters(
            input_table_spec,
            InputColumn.STATE_FEATURES,
            self.model_manager.get_state_preprocessing_options(),
        )

        # Run action feature identification
        action_normalization_parameters = identify_normalization_parameters(
            input_table_spec,
            InputColumn.ACTION,
            self.model_manager.get_action_preprocessing_options(),
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
    def should_generate_eval_dataset(self) -> bool:
        return self.model_manager.eval_parameters.calc_cpe_in_training

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
    ) -> Dataset:
        return data_fetcher.query_data(
            input_table_spec=input_table_spec,
            discrete_action=False,
            include_possible_actions=False,
            custom_reward_expression=reward_options.custom_reward_expression,
            sample_range=sample_range,
        )

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        state_preprocessor = Preprocessor(
            self.state_normalization_data.dense_normalization_parameters,
            use_gpu=self.resource_options.use_gpu,
        )
        action_preprocessor = Preprocessor(
            self.action_normalization_data.dense_normalization_parameters,
            use_gpu=self.resource_options.use_gpu,
        )
        return PolicyNetworkBatchPreprocessor(
            state_preprocessor=state_preprocessor,
            action_preprocessor=action_preprocessor,
            use_gpu=self.resource_options.use_gpu,
        )
