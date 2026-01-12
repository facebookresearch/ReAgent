#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
from typing import Dict, Optional

import torch
from reagent.core import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.model_managers.model_manager import ModelManager
from reagent.models.model_feature_config_provider import RawModelFeatureConfigProvider
from reagent.net_builder.discrete_dqn.dueling import Dueling
from reagent.net_builder.unions import (
    DiscreteDQNNetBuilder__Union,
    ValueNetBuilder__Union,
)
from reagent.training import PPOTrainer, PPOTrainerParameters, ReAgentLightningModule

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import ModelFeatureConfigProvider__Union, RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class PPO(ModelManager):
    __hash__ = param_hash

    trainer_param: PPOTrainerParameters = field(default_factory=PPOTrainerParameters)
    # using DQN net here because it supports `possible_actions_mask`
    policy_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-ignore
        default_factory=lambda: DiscreteDQNNetBuilder__Union(Dueling=Dueling())
    )
    value_net_builder: Optional[ValueNetBuilder__Union] = None
    # pyre-fixme[11]: Annotation `ModelFeatureConfigProvider__Union` is not defined
    #  as a type.
    state_feature_config_provider: ModelFeatureConfigProvider__Union = field(
        # pyre-ignore
        default_factory=lambda: ModelFeatureConfigProvider__Union(
            raw=RawModelFeatureConfigProvider(float_feature_infos=[])
        )
    )
    sampler_temperature: float = 1.0

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self._policy: Optional[Policy] = None
        assert len(self.action_names) > 1, (
            f"PPO needs at least 2 actions. Got {self.action_names}."
        )

    @property
    def action_names(self):
        return self.trainer_param.actions

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        # pyre-fixme[11]: Annotation `RewardOptions` is not defined as a type.
        reward_options: Optional[RewardOptions] = None,
    ) -> PPOTrainer:
        policy_net_builder = self.policy_net_builder.value
        policy_network = policy_net_builder.build_q_network(
            self.state_feature_config,
            normalization_data_map[NormalizationKey.STATE],
            len(self.action_names),
        )
        value_net = None
        value_net_builder = self.value_net_builder
        if value_net_builder:
            value_net_builder = value_net_builder.value
            value_net = value_net_builder.build_value_network(
                normalization_data_map[NormalizationKey.STATE]
            )
        trainer = PPOTrainer(
            policy=self._create_policy(policy_network),
            value_net=value_net,
            **self.trainer_param.asdict(),  # pyre-ignore
        )
        return trainer

    def create_policy(
        self,
        trainer_module: ReAgentLightningModule,
        serving: bool = False,
        normalization_data_map: Optional[Dict[str, NormalizationData]] = None,
    ):
        assert isinstance(trainer_module, PPOTrainer)
        if serving:
            assert normalization_data_map is not None
            return create_predictor_policy_from_model(
                self.build_serving_module(trainer_module, normalization_data_map)
            )
        else:
            return self._create_policy(trainer_module.scorer)

    def _create_policy(self, policy_network):
        if self._policy is None:
            sampler = SoftmaxActionSampler(temperature=self.sampler_temperature)
            self._policy = Policy(scorer=policy_network, sampler=sampler)
        return self._policy

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        assert isinstance(trainer_module, PPOTrainer)
        policy_serving_module = self.policy_net_builder.value.build_serving_module(
            q_network=trainer_module.scorer,
            state_normalization_data=normalization_data_map[NormalizationKey.STATE],
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )
        return policy_serving_module

    @property
    def state_feature_config(self) -> rlt.ModelFeatureConfig:
        return self.state_feature_config_provider.value.get_model_feature_config()
