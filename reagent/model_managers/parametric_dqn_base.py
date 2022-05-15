#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import reagent.core.types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
)
from reagent.data.data_fetcher import DataFetcher
from reagent.data.manual_data_module import ManualDataModule
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.gym.policies.scorers.discrete_scorer import parametric_dqn_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.models.base import ModelBase
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.preprocessing.normalization import get_feature_config
from reagent.preprocessing.types import InputColumn
from reagent.training import ReAgentLightningModule
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
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
            "Please set state allowlist features in state_float_features field of "
            "config instead"
        )
        assert (
            self.action_preprocessing_options is None
            or self.action_preprocessing_options.allowedlist_features is None
        ), (
            "Please set action allowlist features in action_float_features field of "
            "config instead"
        )
        self._q_network: Optional[ModelBase] = None

    def create_policy(
        self,
        trainer_module: ReAgentLightningModule,
        serving: bool = False,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    ):
        """Create an online DiscreteDQN Policy from env."""

        # FIXME: this only works for one-hot encoded actions
        # pyre-fixme[16]: `Tensor` has no attribute `input_prototype`.
        action_dim = trainer_module.q_network.input_prototype()[1].float_features.shape[
            1
        ]
        if serving:
            assert normalization_data_map
            return create_predictor_policy_from_model(
                self.build_serving_module(trainer_module, normalization_data_map),
                max_num_actions=action_dim,
            )
        else:
            # pyre-fixme[16]: `ParametricDQNBase` has no attribute `rl_parameters`.
            sampler = SoftmaxActionSampler(temperature=self.rl_parameters.temperature)
            scorer = parametric_dqn_scorer(
                max_num_actions=action_dim,
                # pyre-fixme[6]: Expected `ModelBase` for 2nd param but got
                #  `Union[torch.Tensor, torch.nn.Module]`.
                q_network=trainer_module.q_network,
            )
            return Policy(scorer=scorer, sampler=sampler)

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.state_float_features)

    @property
    def action_feature_config(self) -> rlt.ModelFeatureConfig:
        return get_feature_config(self.action_float_features)

    # TODO: Add below get_data_module() method once methods in
    # `ParametricDqnDataModule` class are fully implemented
    # def get_data_module(
    #     self,
    #     *,
    #     input_table_spec: Optional[TableSpec] = None,
    #     reward_options: Optional[RewardOptions] = None,
    #     setup_data: Optional[Dict[str, bytes]] = None,
    #     saved_setup_data: Optional[Dict[str, bytes]] = None,
    #     reader_options: Optional[ReaderOptions] = None,
    #     resource_options: Optional[ResourceOptions] = None,
    # ) -> Optional[ReAgentDataModule]:
    #     return ParametricDqnDataModule(
    #         input_table_spec=input_table_spec,
    #         reward_options=reward_options,
    #         setup_data=setup_data,
    #         saved_setup_data=saved_setup_data,
    #         reader_options=reader_options,
    #         resource_options=resource_options,
    #         model_manager=self,
    #     )


class ParametricDqnDataModule(ManualDataModule):
    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.model_manager.eval_parameters.calc_cpe_in_training

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        # Run state feature identification
        state_preprocessing_options = (
            self.model_manager.state_preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id
            for ffi in self.model_manager.state_feature_config.float_feature_infos
        ]
        logger.info(f"state allowedlist_features: {state_features}")
        state_preprocessing_options = replace(
            state_preprocessing_options, allowedlist_features=state_features
        )

        state_normalization_parameters = identify_normalization_parameters(
            input_table_spec, InputColumn.STATE_FEATURES, state_preprocessing_options
        )

        # Run action feature identification
        action_preprocessing_options = (
            self.model_manager.action_preprocessing_options or PreprocessingOptions()
        )
        action_features = [
            ffi.feature_id
            for ffi in self.model_manager.action_feature_config.float_feature_infos
        ]
        logger.info(f"action allowedlist_features: {action_features}")
        action_preprocessing_options = replace(
            action_preprocessing_options, allowedlist_features=action_features
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

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
    ) -> Dataset:
        raise NotImplementedError

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        raise NotImplementedError()
