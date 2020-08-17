#!/usr/bin/env python3

import logging
from typing import Dict

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.types import RewardOptions
from reagent.net_builder.parametric_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import ParametricDQNNetBuilder__Union
from reagent.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.preprocessing.normalization import (
    get_feature_config,
    get_num_output_features,
)
from reagent.training import ParametricDQNTrainer, ParametricDQNTrainerParameters
from reagent.workflow.model_managers.parametric_dqn_base import ParametricDQNBase


logger = logging.getLogger(__name__)


@dataclass
class ParametricDQN(ParametricDQNBase):
    __hash__ = param_hash

    trainer_param: ParametricDQNTrainerParameters = field(
        default_factory=ParametricDQNTrainerParameters
    )
    net_builder: ParametricDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ParametricDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    def build_trainer(
        self,
        use_gpu: bool,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> ParametricDQNTrainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
        # Metrics + reward
        reward_output_dim = len(self.metrics_to_score(reward_options)) + 1
        reward_network = net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
            output_dim=reward_output_dim,
        )

        if use_gpu:
            q_network = q_network.cuda()
            reward_network = reward_network.cuda()

        q_network_target = q_network.get_target_network()
        trainer = ParametricDQNTrainer(
            q_network=q_network,
            q_network_target=q_network_target,
            reward_network=reward_network,
            use_gpu=use_gpu,
            # pyre-fixme[16]: `ParametricDQNTrainerParameters` has no attribute
            #  `asdict`.
            # pyre-fixme[16]: `ParametricDQNTrainerParameters` has no attribute
            #  `asdict`.
            **self.trainer_param.asdict(),
        )

        # HACK: injecting num_actions to build policies for gym
        trainer.num_gym_actions = get_num_output_features(
            normalization_data_map[
                NormalizationKey.ACTION
            ].dense_normalization_parameters
        )

        return trainer

    def build_serving_module(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        trainer: ParametricDQNTrainer,
    ) -> torch.nn.Module:
        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            trainer.q_network,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
