#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import reagent.core.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
    param_hash,
)
from reagent.data.data_fetcher import DataFetcher
from reagent.data.manual_data_module import ManualDataModule
from reagent.data.reagent_data_module import ReAgentDataModule
from reagent.model_managers.model_manager import ModelManager
from reagent.net_builder.synthetic_reward.single_step_synthetic_reward import (
    SingleStepSyntheticReward,
)
from reagent.net_builder.unions import SyntheticRewardNetBuilder__Union
from reagent.preprocessing.types import InputColumn
from reagent.reporting.reward_network_reporter import RewardNetworkReporter
from reagent.training import (
    ReAgentLightningModule,
    RewardNetTrainer,
    RewardNetworkTrainerParameters,
)
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    ResourceOptions,
    RewardOptions,
    TableSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class SyntheticReward(ModelManager):
    """
    Train models to attribute single step rewards from sparse/delayed/aggregated rewards.
    Ideas from:
    1. Synthetic Returns for Long-Term Credit Assignment: https://arxiv.org/pdf/2102.12425.pdf
    2. RUDDER: Return Decomposition for Delayed Rewards: https://arxiv.org/pdf/1806.07857.pdf
    3. Optimizing Agent Behavior over Long Time Scales by Transporting Value: https://arxiv.org/pdf/1810.06721.pdf
    4. Sequence Modeling of Temporal Credit Assignment for Episodic Reinforcement Learning: https://arxiv.org/pdf/1905.13420.pdf
    """

    __hash__ = param_hash

    trainer_param: RewardNetworkTrainerParameters = field(
        default_factory=RewardNetworkTrainerParameters
    )
    net_builder: SyntheticRewardNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `SlateRewardTransformer`.
        default_factory=lambda: SyntheticRewardNetBuilder__Union(
            SingleStepSyntheticReward=SingleStepSyntheticReward()
        )
    )
    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)
    state_preprocessing_options: Optional[PreprocessingOptions] = None
    action_preprocessing_options: Optional[PreprocessingOptions] = None
    state_feature_config: rlt.ModelFeatureConfig = field(
        default_factory=rlt.ModelFeatureConfig
    )
    parametric_action_feature_config: rlt.ModelFeatureConfig = field(
        default_factory=rlt.ModelFeatureConfig
    )
    discrete_action_names: Optional[List[str]] = None
    # max sequence length to look back to distribute rewards
    max_seq_len: int = 5

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        assert self.max_seq_len is not None and self.max_seq_len > 0
        assert (
            self.state_preprocessing_options is None
            or self.state_preprocessing_options.allowedlist_features is None
        ), (
            "Please set state allowlist features in state_float_features field of "
            "config instead"
        )

        if self.discrete_action_names:
            assert (
                type(self.discrete_action_names) is list
                and len(self.discrete_action_names) > 1
            ), f"Assume this is a discrete action problem, you need to specify at least 2 actions. Got {self.discrete_action_names}."
        else:
            assert (
                self.action_preprocessing_options is None
                or self.action_preprocessing_options.allowedlist_features is None
            ), (
                "Please set action allowlist features in parametric_action_float_features field of "
                "config instead"
            )

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
        return SyntheticRewardDataModule(
            input_table_spec=input_table_spec,
            reward_options=reward_options,
            setup_data=setup_data,
            saved_setup_data=saved_setup_data,
            reader_options=reader_options,
            resource_options=resource_options,
            model_manager=self,
        )

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        reward_options: Optional[RewardOptions] = None,
    ) -> RewardNetTrainer:
        net_builder = self.net_builder.value
        action_normalization_data = None
        if not self.discrete_action_names:
            action_normalization_data = normalization_data_map[NormalizationKey.ACTION]
        synthetic_reward_network = net_builder.build_synthetic_reward_network(
            normalization_data_map[NormalizationKey.STATE],
            action_normalization_data=action_normalization_data,
            discrete_action_names=self.discrete_action_names,
            state_feature_config=self.state_feature_config,
            action_feature_config=self.parametric_action_feature_config,
        )

        trainer = RewardNetTrainer(
            synthetic_reward_network,
            # pyre-fixme[16]: `RewardNetworkTrainerParameters` has no attribute
            #  `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def get_reporter(self):
        return RewardNetworkReporter(
            self.trainer_param.loss_type,
            str(self.net_builder.value),
        )

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        assert isinstance(trainer_module, RewardNetTrainer)

        net_builder = self.net_builder.value
        action_normalization_data = None
        if not self.discrete_action_names:
            action_normalization_data = normalization_data_map[NormalizationKey.ACTION]
        return net_builder.build_serving_module(
            self.max_seq_len,
            trainer_module.reward_net,
            normalization_data_map[NormalizationKey.STATE],
            action_normalization_data=action_normalization_data,
            discrete_action_names=self.discrete_action_names,
            state_feature_config=self.state_feature_config,
            action_feature_config=self.parametric_action_feature_config,
        )


class SyntheticRewardDataModule(ManualDataModule):
    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.model_manager.eval_parameters.calc_cpe_in_training

    def run_feature_identification(
        self, input_table_spec: TableSpec
    ) -> Dict[str, NormalizationData]:
        """Identify dense feature normalization parameters"""
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
        if self.model_manager.discrete_action_names:
            return {
                NormalizationKey.STATE: NormalizationData(
                    dense_normalization_parameters=state_normalization_parameters
                )
            }
        # Run action feature identification
        action_preprocessing_options = (
            self.model_manager.action_preprocessing_options or PreprocessingOptions()
        )
        action_features = [
            ffi.feature_id
            for ffi in self.model_manager.parametric_action_feature_config.float_feature_infos
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
        return data_fetcher.query_data_synthetic_reward(
            input_table_spec=input_table_spec,
            discrete_action_names=self.model_manager.discrete_action_names,
            sample_range=sample_range,
            max_seq_len=self.model_manager.max_seq_len,
        )

    def build_batch_preprocessor(self):
        raise NotImplementedError
