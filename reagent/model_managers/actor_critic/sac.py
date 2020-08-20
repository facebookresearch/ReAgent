#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.rl_training_output import RLTrainingOutput
from reagent.core.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
    TableSpec,
)
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
from reagent.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.training import SACTrainer, SACTrainerParameters


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

    def build_trainer(
        self,
        use_gpu: bool,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> SACTrainer:
        actor_net_builder = self.actor_net_builder.value
        actor_network = actor_net_builder.build_actor(
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
        if self.value_net_builder:
            # pyre-fixme[16]: `Optional` has no attribute `value`.
            # pyre-fixme[16]: `Optional` has no attribute `value`.
            value_net_builder = self.value_net_builder.value
            value_network = value_net_builder.build_value_network(
                normalization_data_map[NormalizationKey.STATE]
            )

        if use_gpu:
            q1_network.cuda()
            if q2_network:
                q2_network.cuda()
            if value_network:
                value_network.cuda()
            actor_network.cuda()

        trainer = SACTrainer(
            actor_network=actor_network,
            q1_network=q1_network,
            value_network=value_network,
            q2_network=q2_network,
            use_gpu=use_gpu,
            # pyre-fixme[16]: `SACTrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `SACTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def build_serving_module(
        self, normalization_data_map: Dict[str, NormalizationData], trainer: SACTrainer
    ) -> torch.nn.Module:
        net_builder = self.actor_net_builder.value
        return net_builder.build_serving_module(
            trainer.actor_network,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
            serve_mean_policy=self.serve_mean_policy,
        )
