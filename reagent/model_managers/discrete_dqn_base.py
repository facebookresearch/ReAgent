#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import abc
import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from reagent.core import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
    RLParameters,
)
from reagent.data.data_fetcher import DataFetcher
from reagent.data.manual_data_module import ManualDataModule
from reagent.data.reagent_data_module import ReAgentDataModule
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.gym.policies.scorers.discrete_scorer import discrete_dqn_scorer
from reagent.model_managers.model_manager import ModelManager
from reagent.models.model_feature_config_provider import RawModelFeatureConfigProvider
from reagent.preprocessing.batch_preprocessor import (
    BatchPreprocessor,
    DiscreteDqnBatchPreprocessor,
)
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.preprocessing.types import InputColumn
from reagent.reporting.discrete_dqn_reporter import DiscreteDQNReporter
from reagent.training import ReAgentLightningModule

# pyre-fixme[21]: Could not find module `reagent.workflow.identify_types_flow`.
from reagent.workflow.identify_types_flow import identify_normalization_parameters

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import (
    Dataset,
    ModelFeatureConfigProvider__Union,
    PreprocessingOptions,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    TableSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscreteDQNBase(ModelManager):
    target_action_distribution: Optional[List[float]] = None
    # pyre-fixme[11]: Annotation `ModelFeatureConfigProvider__Union` is not defined
    #  as a type.
    state_feature_config_provider: ModelFeatureConfigProvider__Union = field(
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        default_factory=lambda: ModelFeatureConfigProvider__Union(
            raw=RawModelFeatureConfigProvider(float_feature_infos=[])
        )
    )
    # pyre-fixme[11]: Annotation `PreprocessingOptions` is not defined as a type.
    preprocessing_options: Optional[PreprocessingOptions] = None
    # pyre-fixme[11]: Annotation `ReaderOptions` is not defined as a type.
    reader_options: Optional[ReaderOptions] = None
    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    @property
    @abc.abstractmethod
    def rl_parameters(self) -> RLParameters:
        pass

    @property
    @abc.abstractmethod
    def action_names(self) -> List[str]:
        # Returns the list of possible actions for this instance of problem
        pass

    def create_policy(
        self,
        trainer_module: ReAgentLightningModule,
        serving: bool = False,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    ) -> Policy:
        """Create an online DiscreteDQN Policy from env."""
        if serving:
            assert normalization_data_map
            return create_predictor_policy_from_model(
                self.build_serving_module(trainer_module, normalization_data_map),
                rl_parameters=self.rl_parameters,
            )
        else:
            sampler = GreedyActionSampler()
            # pyre-fixme[6]: For 1st argument expected `ModelBase` but got
            #  `Union[Tensor, Module]`.
            scorer = discrete_dqn_scorer(trainer_module.q_network)
            return Policy(scorer=scorer, sampler=sampler)

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return self.state_feature_config_provider.value.get_model_feature_config()

    def get_state_preprocessing_options(self) -> PreprocessingOptions:
        state_preprocessing_options = (
            # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
            self.preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id for ffi in self.state_feature_config.float_feature_infos
        ]
        logger.info(f"state allowedlist_features: {state_features}")
        state_preprocessing_options = replace(
            state_preprocessing_options, allowedlist_features=state_features
        )
        return state_preprocessing_options

    @property
    def multi_steps(self) -> Optional[int]:
        return self.rl_parameters.multi_steps

    def get_data_module(
        self,
        *,
        # pyre-fixme[11]: Annotation `TableSpec` is not defined as a type.
        input_table_spec: Optional[TableSpec] = None,
        # pyre-fixme[11]: Annotation `RewardOptions` is not defined as a type.
        reward_options: Optional[RewardOptions] = None,
        reader_options: Optional[ReaderOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        # pyre-fixme[11]: Annotation `ResourceOptions` is not defined as a type.
        resource_options: Optional[ResourceOptions] = None,
    ) -> Optional[ReAgentDataModule]:
        return DiscreteDqnDataModule(
            input_table_spec=input_table_spec,
            reward_options=reward_options,
            setup_data=setup_data,
            saved_setup_data=saved_setup_data,
            reader_options=reader_options,
            resource_options=resource_options,
            model_manager=self,
        )

    def get_reporter(self):
        return DiscreteDQNReporter(
            self.trainer_param.actions,
            target_action_distribution=self.target_action_distribution,
        )


class DiscreteDqnDataModule(ManualDataModule):
    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.model_manager.eval_parameters.calc_cpe_in_training

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        preprocessing_options = (
            # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
            self.model_manager.preprocessing_options or PreprocessingOptions()
        )
        state_features = [
            ffi.feature_id
            for ffi in self.model_manager.state_feature_config.float_feature_infos
        ]
        logger.info(f"Overriding allowedlist_features: {state_features}")
        preprocessing_options = replace(
            preprocessing_options, allowedlist_features=state_features
        )
        return {
            NormalizationKey.STATE: NormalizationData(
                # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
                dense_normalization_parameters=identify_normalization_parameters(
                    input_table_spec, InputColumn.STATE_FEATURES, preprocessing_options
                )
            )
        }

    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        reward_options: RewardOptions,
        data_fetcher: DataFetcher,
        # pyre-fixme[11]: Annotation `Dataset` is not defined as a type.
    ) -> Dataset:
        return data_fetcher.query_data(
            input_table_spec=input_table_spec,
            discrete_action=True,
            actions=self.model_manager.action_names,
            include_possible_actions=True,
            sample_range=sample_range,
            custom_reward_expression=reward_options.custom_reward_expression,
            multi_steps=self.model_manager.multi_steps,
            gamma=self.model_manager.rl_parameters.gamma,
        )

    def build_batch_preprocessor(self) -> BatchPreprocessor:
        state_preprocessor = Preprocessor(
            self.state_normalization_data.dense_normalization_parameters,
        )
        return DiscreteDqnBatchPreprocessor(
            num_actions=len(self.model_manager.action_names),
            state_preprocessor=state_preprocessor,
        )
