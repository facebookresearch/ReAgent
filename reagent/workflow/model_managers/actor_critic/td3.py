#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.types import (
    Dataset,
    PreprocessingOptions,
    ReaderOptions,
    RewardOptions,
    RLTrainingOutput,
    TableSpec,
)
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
from reagent.parameters import (
    EvaluationParameters,
    NormalizationData,
    NormalizationKey,
    param_hash,
)
from reagent.training import TD3Trainer, TD3TrainerParameters
from reagent.workflow.model_managers.actor_critic_base import ActorCriticBase


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

    def build_trainer(
        self,
        use_gpu: bool,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> TD3Trainer:
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

        if use_gpu:
            q1_network.cuda()
            if q2_network:
                q2_network.cuda()
            actor_network.cuda()

        trainer = TD3Trainer(
            actor_network=actor_network,
            q1_network=q1_network,
            q2_network=q2_network,
            use_gpu=use_gpu,
            # pyre-fixme[16]: `TD3TrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `TD3TrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

    def build_serving_module(
        self, normalization_data_map: Dict[str, NormalizationData], trainer: TD3Trainer
    ) -> torch.nn.Module:
        net_builder = self.actor_net_builder.value
        return net_builder.build_serving_module(
            trainer.actor_network,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
