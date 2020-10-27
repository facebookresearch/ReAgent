#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.models.base import ModelBase
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
from reagent.parameters import EvaluationParameters, param_hash
from reagent.training import TD3Trainer, TD3TrainerParameters
from reagent.workflow.model_managers.actor_critic_base import ActorCriticBase
from reagent.workflow.reporters.td3_reporter import TD3Reporter


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
    use_2_q_functions: bool = True
    eval_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self._actor_network: Optional[ModelBase] = None
        self.rl_parameters = self.trainer_param.rl

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(self) -> TD3Trainer:
        actor_net_builder = self.actor_net_builder.value
        # pyre-fixme[16]: `TD3` has no attribute `_actor_network`.
        # pyre-fixme[16]: `TD3` has no attribute `_actor_network`.
        self._actor_network = actor_net_builder.build_actor(
            self.state_normalization_data, self.action_normalization_data
        )

        critic_net_builder = self.critic_net_builder.value
        # pyre-fixme[16]: `TD3` has no attribute `_q1_network`.
        # pyre-fixme[16]: `TD3` has no attribute `_q1_network`.
        self._q1_network = critic_net_builder.build_q_network(
            self.state_normalization_data, self.action_normalization_data
        )
        q2_network = (
            critic_net_builder.build_q_network(
                self.state_normalization_data, self.action_normalization_data
            )
            if self.use_2_q_functions
            else None
        )

        trainer = TD3Trainer(
            actor_network=self._actor_network,
            q1_network=self._q1_network,
            q2_network=q2_network,
            # pyre-fixme[16]: `TD3TrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `TD3TrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def get_reporter(self):
        return TD3Reporter()

    def build_serving_module(self) -> torch.nn.Module:
        net_builder = self.actor_net_builder.value
        assert self._actor_network is not None
        return net_builder.build_serving_module(
            self._actor_network,
            self.state_normalization_data,
            self.action_normalization_data,
        )
