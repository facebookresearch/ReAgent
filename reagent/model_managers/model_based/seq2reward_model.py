#!/usr/bin/env python3

import logging
from typing import Optional, Dict

from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import (
    Seq2RewardTrainerParameters,
    param_hash,
    NormalizationKey,
    NormalizationData,
)
from reagent.model_managers.world_model_base import WorldModelBase
from reagent.net_builder.unions import ValueNetBuilder__Union
from reagent.net_builder.value.fully_connected import FullyConnected
from reagent.net_builder.value.seq2reward_rnn import Seq2RewardNetBuilder
from reagent.reporting.seq2reward_reporter import Seq2RewardReporter
from reagent.training.world_model.seq2reward_trainer import Seq2RewardTrainer
from reagent.workflow.types import PreprocessingOptions, RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class Seq2RewardModel(WorldModelBase):
    __hash__ = param_hash
    net_builder: ValueNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `Seq2RewardNetBuilder`.
        # pyre-fixme[28]: Unexpected keyword argument `Seq2RewardNetBuilder`.
        default_factory=lambda: ValueNetBuilder__Union(
            Seq2RewardNetBuilder=Seq2RewardNetBuilder()
        )
    )

    compress_net_builder: ValueNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        default_factory=lambda: ValueNetBuilder__Union(FullyConnected=FullyConnected())
    )

    trainer_param: Seq2RewardTrainerParameters = field(
        default_factory=Seq2RewardTrainerParameters
    )

    preprocessing_options: Optional[PreprocessingOptions] = None

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        reward_options: Optional[RewardOptions] = None,
    ) -> Seq2RewardTrainer:
        # pyre-fixme[16]: `Seq2RewardModel` has no attribute `_seq2reward_network`.
        self._seq2reward_network = (
            seq2reward_network
        ) = self.net_builder.value.build_value_network(
            normalization_data_map[NormalizationKey.STATE]
        )
        trainer = Seq2RewardTrainer(
            seq2reward_network=seq2reward_network, params=self.trainer_param
        )
        # pyre-fixme[16]: `Seq2RewardModel` has no attribute `_step_predict_network`.
        self._step_predict_network = trainer.step_predict_network
        return trainer

    def get_reporter(self) -> Seq2RewardReporter:
        return Seq2RewardReporter(self.trainer_param.action_names)
