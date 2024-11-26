#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
from typing import Dict, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, NormalizationKey, param_hash
from reagent.model_managers.slate_q_base import SlateQBase
from reagent.net_builder.parametric_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import ParametricDQNNetBuilder__Union
from reagent.training import (
    ReAgentLightningModule,
    SlateQTrainer,
    SlateQTrainerParameters,
)

# pyre-fixme[21]: Could not find module `reagent.workflow.types`.
from reagent.workflow.types import RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class SlateQ(SlateQBase):
    __hash__ = param_hash

    slate_size: int = -1
    num_candidates: int = -1
    trainer_param: SlateQTrainerParameters = field(
        default_factory=SlateQTrainerParameters
    )
    net_builder: ParametricDQNNetBuilder__Union = field(
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
        self.eval_parameters = self.trainer_param.evaluation

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        # pyre-fixme[11]: Annotation `RewardOptions` is not defined as a type.
        reward_options: Optional[RewardOptions] = None,
    ) -> SlateQTrainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ITEM],
        )

        q_network_target = q_network.get_target_network()
        return SlateQTrainer(
            q_network=q_network,
            q_network_target=q_network_target,
            slate_size=self.slate_size,
            # pyre-fixme[16]: `SlateQTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        assert isinstance(trainer_module, SlateQTrainer)
        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            trainer_module.q_network,
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ITEM],
        )
