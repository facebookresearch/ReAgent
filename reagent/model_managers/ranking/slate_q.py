#!/usr/bin/env python3

import logging
from typing import Optional, Dict

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import param_hash, NormalizationData, NormalizationKey
from reagent.model_managers.slate_q_base import SlateQBase
from reagent.models.base import ModelBase
from reagent.net_builder.parametric_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import ParametricDQNNetBuilder__Union
from reagent.training import SlateQTrainer, SlateQTrainerParameters


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
        self._q_network: Optional[ModelBase] = None
        self.eval_parameters = self.trainer_param.evaluation

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(
        self, normalization_data_map: Dict[str, NormalizationData], use_gpu: bool
    ) -> SlateQTrainer:
        net_builder = self.net_builder.value
        # pyre-fixme[16]: `SlateQ` has no attribute `_q_network`.
        self._q_network = net_builder.build_q_network(
            normalization_data_map[NormalizationKey.STATE],
            normalization_data_map[NormalizationKey.ITEM],
        )

        q_network_target = self._q_network.get_target_network()
        return SlateQTrainer(
            q_network=self._q_network,
            q_network_target=q_network_target,
            # pyre-fixme[16]: `SlateQTrainerParameters` has no attribute `asdict`.
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
            normalization_data_map[NormalizationKey.ITEM],
        )
