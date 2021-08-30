#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.model_managers.actor_critic_base import ActorCriticBase
from reagent.models.base import ModelBase
from reagent.net_builder.continuous_actor.gaussian_fully_connected import (
    GaussianFullyConnected,
)
from reagent.net_builder.parametric_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import (
    ContinuousActorNetBuilder__Union,
    ParametricDQNNetBuilder__Union,
    ValueNetBuilder__Union,
)
from reagent.net_builder.value.fully_connected import (
    FullyConnected as ValueFullyConnected,
)
from reagent.training import ReAgentLightningModule
from reagent.training import SACTrainer, SACTrainerParameters
from reagent.workflow.types import RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class SAC(ActorCriticBase):
    __hash__ = param_hash

    trainer_param: SACTrainerParameters = field(default_factory=SACTrainerParameters)
    actor_net_builder: ContinuousActorNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `GaussianFullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `GaussianFullyConnected`.
        default_factory=lambda: ContinuousActorNetBuilder__Union(
            GaussianFullyConnected=GaussianFullyConnected()
        )
    )
    critic_net_builder: ParametricDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ParametricDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )
    value_net_builder: Optional[ValueNetBuilder__Union] = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ValueNetBuilder__Union(
            FullyConnected=ValueFullyConnected()
        )
    )
    use_2_q_functions: bool = True
    serve_mean_policy: bool = True

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self.rl_parameters = self.trainer_param.rl

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        reward_options: Optional[RewardOptions] = None,
    ) -> SACTrainer:
        actor_net_builder = self.actor_net_builder.value
        actor_network = actor_net_builder.build_actor(
            self.state_feature_config,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )

        critic_net_builder = self.critic_net_builder.value
        q1_network = critic_net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
        q2_network = (
            critic_net_builder.build_q_network(
                normalization_data_map[NormalizationKey.STATE],
                normalization_data_map[NormalizationKey.ACTION],
            )
            if self.use_2_q_functions
            else None
        )

        value_network = None
        value_net_builder = self.value_net_builder
        if value_net_builder:
            value_net_builder = value_net_builder.value
            value_network = value_net_builder.build_value_network(
                normalization_data_map[NormalizationKey.STATE]
            )

        trainer = SACTrainer(
            actor_network=actor_network,
            q1_network=q1_network,
            value_network=value_network,
            q2_network=q2_network,
            # pyre-fixme[16]: `SACTrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `SACTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def get_reporter(self):
        return None

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        assert isinstance(trainer_module, SACTrainer)
        actor_serving_module = self.actor_net_builder.value.build_serving_module(
            trainer_module.actor_network,
            self.state_feature_config,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
            serve_mean_policy=self.serve_mean_policy,
        )
        return actor_serving_module

    # TODO: add in critic
    # assert self._q1_network is not None
    # _critic_serving_module = self.critic_net_builder.value.build_serving_module(
    #     self._q1_network,
    #     self.state_normalization_data,
    #     self.action_normalization_data,
    # )
