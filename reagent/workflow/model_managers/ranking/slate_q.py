#!/usr/bin/env python3

import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.types import RewardOptions
from reagent.models.base import ModelBase
from reagent.net_builder.parametric_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import ParametricDQNNetBuilder__Union
from reagent.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.training import SlateQTrainer, SlateQTrainerParameters
from reagent.workflow.model_managers.slate_q_base import SlateQBase


logger = logging.getLogger(__name__)


@dataclass
class SlateQ(SlateQBase):
    __hash__ = param_hash

    net_builder: ParametricDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ParametricDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        assert (
            self.slate_size > 0
        ), f"Please set valid slate_size (currently {self.slate_size})"
        assert (
            self.num_candidates > 0
        ), f"Please set valid num_candidates (currently {self.num_candidates})"

    def build_trainer(
        self,
        use_gpu: bool,
        normalization_data_map: Dict[str, NormalizationData],
        reward_options: RewardOptions,
    ) -> SlateQTrainer:
        net_builder = self.net_builder.value
        # pyre-fixme[16]: `SlateQ` has no attribute `_q_network`.
        # pyre-fixme[16]: `SlateQ` has no attribute `_q_network`.
        q_network = net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ITEM],
        )
        if use_gpu:
            q_network = q_network.cuda()

        q_network_target = q_network.get_target_network()
        return SlateQTrainer(
            q_network=q_network,
            q_network_target=q_network_target,
            use_gpu=use_gpu,
            # pyre-fixme[16]: `SlateQTrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `SlateQTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )

    def build_serving_module(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        trainer: SlateQTrainer,
    ) -> torch.nn.Module:
        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            trainer.q_network,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ITEM],
        )
