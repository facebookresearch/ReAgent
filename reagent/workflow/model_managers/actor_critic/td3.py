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
from reagent.parameters import EvaluationParameters, NormalizationKey, param_hash
from reagent.training import TD3Trainer, TD3TrainingParameters
from reagent.workflow.model_managers.actor_critic_base import ActorCriticBase


logger = logging.getLogger(__name__)


@dataclass
class TD3(ActorCriticBase):
    __hash__ = param_hash

    trainer_param: TD3TrainingParameters = field(default_factory=TD3TrainingParameters)
    actor_net_builder: ContinuousActorNetBuilder__Union = field(
        default_factory=lambda: ContinuousActorNetBuilder__Union(
            FullyConnected=ContinuousFullyConnected()
        )
    )
    critic_net_builder: ParametricDQNNetBuilder__Union = field(
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

    def build_trainer(self) -> TD3Trainer:
        actor_net_builder = self.actor_net_builder.value
        self._actor_network = actor_net_builder.build_actor(
            self.get_normalization_data(NormalizationKey.STATE),
            self.get_normalization_data(NormalizationKey.ACTION),
        )

        critic_net_builder = self.critic_net_builder.value
        q1_network = critic_net_builder.build_q_network(
            self.state_normalization_parameters, self.action_normalization_parameters
        )
        q2_network = (
            critic_net_builder.build_q_network(
                self.state_normalization_parameters,
                self.action_normalization_parameters,
            )
            if self.use_2_q_functions
            else None
        )

        if self.use_gpu:
            q1_network.cuda()
            if q2_network:
                q2_network.cuda()
            self._actor_network.cuda()

        trainer = TD3Trainer(
            q1_network,
            self._actor_network,
            self.trainer_param,
            q2_network=q2_network,
            use_gpu=self.use_gpu,
        )
        return trainer

    def build_serving_module(self) -> torch.nn.Module:
        net_builder = self.actor_net_builder.value
        assert self._actor_network is not None
        return net_builder.build_serving_module(
            self._actor_network,
            self.get_normalization_data(NormalizationKey.STATE),
            self.get_normalization_data(NormalizationKey.ACTION),
        )
