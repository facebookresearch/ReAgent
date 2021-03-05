#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.models.base import ModelBase
from reagent.parameters import EvaluationParameters, NormalizationData, NormalizationKey
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    PolicyNetworkBatchPreprocessor,
    Preprocessor,
)
from reagent.preprocessing.normalization import get_feature_config
from reagent.preprocessing.types import InputColumn
from reagent.workflow.data import ReAgentDataModule
from reagent.workflow.data_fetcher import query_data
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.model_managers.model_manager import ModelManager
from reagent.workflow.reporters.actor_critic_reporter import ActorCriticReporter
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    RLTrainingOutput,
    RLTrainingReport,
    TableSpec,
)
from reagent.workflow.utils import train_eval_lightning


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
    save_critic_bool: bool = True

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
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
        self._state_preprocessing_options = self.state_preprocessing_options
        self._action_preprocessing_options = self.action_preprocessing_options

        # To be filled by property metrics_to_score
        self._metrics_to_score: Optional[List[str]] = None

        # To be filled by subclasses
        self._actor_network: Optional[ModelBase] = None
        self._q1_network: Optional[ModelBase] = None

    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.eval_parameters.calc_cpe_in_training

    def create_policy(self, serving: bool) -> Policy:
        """ Create online actor critic policy. """

        if serving:
            return create_predictor_policy_from_model(self.build_serving_module())
        else:
            return ActorPolicyWrapper(self._actor_network)

    @property
    def metrics_to_score(self) -> List[str]:
        assert self._reward_options is not None
        if self._metrics_to_score is None:
            # pyre-fixme[16]: `ActorCriticBase` has no attribute `_metrics_to_score`.
            # pyre-fixme[16]: `ActorCriticBase` has no attribute `_metrics_to_score`.
            self._metrics_to_score = get_metrics_to_score(
                self._reward_options.metric_reward_values
            )
        return self._metrics_to_score

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def action_feature_config(self) -> rlt.ModelFeatureConfig:
        assert len(self.action_float_features) > 0, "You must set action_float_features"
        return get_feature_config(self.action_float_features)

    def get_state_preprocessing_options(self) -> PreprocessingOptions:
        state_preprocessing_options = (
            self._state_preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id for ffi in self.state_feature_config.float_feature_infos
        ]
        logger.info(f"state whitelist_features: {state_features}")
        state_preprocessing_options = state_preprocessing_options._replace(
            whitelist_features=state_features
        )
        return state_preprocessing_options

    def get_action_preprocessing_options(self) -> PreprocessingOptions:
        action_preprocessing_options = (
            self._action_preprocessing_options or PreprocessingOptions()
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
        return action_preprocessing_options

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        # Run state feature identification
        state_normalization_parameters = identify_normalization_parameters(
            input_table_spec,
            InputColumn.STATE_FEATURES,
            self.get_state_preprocessing_options(),
        )

        # Run action feature identification
        action_normalization_parameters = identify_normalization_parameters(
            input_table_spec,
            InputColumn.ACTION,
            self.get_action_preprocessing_options(),
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
    ) -> Dataset:
        logger.info("Starting query")
        return query_data(
            input_table_spec=input_table_spec,
            discrete_action=False,
            include_possible_actions=False,
            custom_reward_expression=reward_options.custom_reward_expression,
            sample_range=sample_range,
        )

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        state_preprocessor = Preprocessor(
            self.state_normalization_data.dense_normalization_parameters,
            use_gpu=self.use_gpu,
        )
        action_preprocessor = Preprocessor(
            self.action_normalization_data.dense_normalization_parameters,
            use_gpu=self.use_gpu,
        )
        return PolicyNetworkBatchPreprocessor(
            state_preprocessor=state_preprocessor,
            action_preprocessor=action_preprocessor,
            use_gpu=self.use_gpu,
        )

    def get_reporter(self):
        return ActorCriticReporter()

    # TODO: deprecate, once we deprecate internal page handlers
    def train(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        data_module: Optional[ReAgentDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: Optional[ResourceOptions],
    ) -> RLTrainingOutput:

        batch_preprocessor = self.build_batch_preprocessor()
        reporter = self.get_reporter()
        # pyre-fixme[16]: `Trainer` has no attribute `set_reporter`.
        # pyre-fixme[16]: `Trainer` has no attribute `set_reporter`.
        self.trainer.set_reporter(reporter)

        # assert eval_dataset is None

        # pyre-fixme[16]: `ActorCriticBase` has no attribute `_lightning_trainer`.
        self._lightning_trainer = train_eval_lightning(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=None,
            trainer_module=self.trainer,
            data_module=data_module,
            num_epochs=num_epochs,
            use_gpu=self.use_gpu,
            batch_preprocessor=batch_preprocessor,
            reader_options=self.reader_options,
            checkpoint_path=self._lightning_checkpoint_path,
            resource_options=resource_options or ResourceOptions(),
        )
        # pyre-fixme[16]: `RLTrainingReport` has no attribute `make_union_instance`.
        training_report = RLTrainingReport.make_union_instance(
            reporter.generate_training_report()
        )
        return RLTrainingOutput(training_report=training_report)
