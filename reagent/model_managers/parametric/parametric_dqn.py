#!/usr/bin/env python3

import logging
from typing import Dict

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import param_hash, NormalizationData, NormalizationKey
from reagent.model_managers.parametric_dqn_base import ParametricDQNBase
from reagent.net_builder.parametric_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import ParametricDQNNetBuilder__Union
from reagent.training import ParametricDQNTrainer, ParametricDQNTrainerParameters


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

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self.rl_parameters = self.trainer_param.rl

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(self, use_gpu: bool) -> ParametricDQNTrainer:
        net_builder = self.net_builder.value
        # pyre-fixme[16]: `ParametricDQN` has no attribute `_q_network`.
        self._q_network = net_builder.build_q_network(
            self.state_normalization_data, self.action_normalization_data
        )
        # Metrics + reward
        reward_output_dim = len(self.metrics_to_score) + 1
        reward_network = net_builder.build_q_network(
            self.state_normalization_data,
            self.action_normalization_data,
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
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        net_builder = self.net_builder.value
        assert self._q_network is not None
        return net_builder.build_serving_module(
            self._q_network,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ACTION],
        )
