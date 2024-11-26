#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.model_managers.parametric_dqn_base import ParametricDQNBase
from reagent.net_builder.parametric_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import ParametricDQNNetBuilder__Union
from reagent.training import (
    ParametricDQNTrainer,
    ParametricDQNTrainerParameters,
    ReAgentLightningModule,
)

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class ParametricDQN(ParametricDQNBase):
    __hash__ = param_hash

    trainer_param: ParametricDQNTrainerParameters = field(
        default_factory=ParametricDQNTrainerParameters
    )
    net_builder: ParametricDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ParametricDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    @property
    def rl_parameters(self):
        return self.trainer_param.rl

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        # pyre-fixme[11]: Annotation `RewardOptions` is not defined as a type.
        reward_options: Optional[RewardOptions] = None,
    ) -> ParametricDQNTrainer:
        net_builder = self.net_builder.value
        # pyre-fixme[16]: `ParametricDQN` has no attribute `_q_network`.
        self._q_network = net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
        # Metrics + reward
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        reward_options = reward_options or RewardOptions()
        metrics_to_score = get_metrics_to_score(reward_options.metric_reward_values)
        reward_output_dim = len(metrics_to_score) + 1
        reward_network = net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
            output_dim=reward_output_dim,
        )

        q_network_target = self._q_network.get_target_network()
        return ParametricDQNTrainer(
            q_network=self._q_network,
            q_network_target=q_network_target,
            reward_network=reward_network,
            # pyre-fixme[16]: `ParametricDQNTrainerParameters` has no attribute
            #  `asdict`.
            **self.trainer_param.asdict(),
        )

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        assert isinstance(trainer_module, ParametricDQNTrainer)
        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            trainer_module.q_network,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
