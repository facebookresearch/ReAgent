#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe


import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
    param_hash,
)
from reagent.model_managers.actor_critic_base import ActorCriticBase
from reagent.net_builder.continuous_actor.fully_connected import (
    FullyConnected as ContinuousFullyConnected,
)
from reagent.net_builder.parametric_dqn.fully_connected import (
    FullyConnected as ParametricFullyConnected,
)
from reagent.net_builder.unions import (
    ContinuousActorNetBuilder__Union,
    ParametricDQNNetBuilder__Union,
)
from reagent.reporting.td3_reporter import TD3Reporter
from reagent.training import ReAgentLightningModule, TD3Trainer, TD3TrainerParameters

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class TD3(ActorCriticBase):
    __hash__ = param_hash

    trainer_param: TD3TrainerParameters = field(default_factory=TD3TrainerParameters)
    actor_net_builder: ContinuousActorNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ContinuousActorNetBuilder__Union(
            FullyConnected=ContinuousFullyConnected()
        )
    )
    critic_net_builder: ParametricDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ParametricDQNNetBuilder__Union(
            FullyConnected=ParametricFullyConnected()
        )
    )
    # Why isn't this a parameter in the .yaml config file?
    use_2_q_functions: bool = True
    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self.rl_parameters = self.trainer_param.rl

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        # pyre-fixme[11]: Annotation `RewardOptions` is not defined as a type.
        reward_options: Optional[RewardOptions] = None,
    ) -> TD3Trainer:
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

        trainer = TD3Trainer(
            actor_network=actor_network,
            q1_network=q1_network,
            q2_network=q2_network,
            # pyre-fixme[16]: `TD3TrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `TD3TrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def get_reporter(self):
        return TD3Reporter()

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        assert isinstance(trainer_module, TD3Trainer)
        net_builder = self.actor_net_builder.value
        return net_builder.build_serving_module(
            trainer_module.actor_network,
            self.state_feature_config,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
