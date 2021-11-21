#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Dict
from typing import Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import param_hash, NormalizationData, NormalizationKey
from reagent.model_managers.discrete_dqn_base import DiscreteDQNBase
from reagent.net_builder.categorical_dqn.categorical import Categorical
from reagent.net_builder.unions import CategoricalDQNNetBuilder__Union
from reagent.training import C51Trainer, C51TrainerParameters
from reagent.training import ReAgentLightningModule
from reagent.workflow.types import RewardOptions


logger = logging.getLogger(__name__)


@dataclass
class DiscreteC51DQN(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: C51TrainerParameters = field(default_factory=C51TrainerParameters)
    net_builder: CategoricalDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `Categorical`.
        # pyre-fixme[28]: Unexpected keyword argument `Categorical`.
        default_factory=lambda: CategoricalDQNNetBuilder__Union(
            Categorical=Categorical()
        )
    )
    cpe_net_builder: CategoricalDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `Categorical`.
        # pyre-fixme[28]: Unexpected keyword argument `Categorical`.
        default_factory=lambda: CategoricalDQNNetBuilder__Union(
            Categorical=Categorical()
        )
    )

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        assert len(self.action_names) > 1, "DiscreteC51DQN needs at least 2 actions"
        assert (
            self.trainer_param.minibatch_size % 8 == 0
        ), "The minibatch size must be divisible by 8 for performance reasons."

    @property
    def action_names(self):
        return self.trainer_param.actions

    @property
    def rl_parameters(self):
        return self.trainer_param.rl

    def build_trainer(
        self,
        normalization_data_map: Dict[str, NormalizationData],
        use_gpu: bool,
        reward_options: Optional[RewardOptions] = None,
    ) -> C51Trainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            state_normalization_data=normalization_data_map[NormalizationKey.STATE],
            output_dim=len(self.action_names),
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `num_atoms`.
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `num_atoms`.
            num_atoms=self.trainer_param.num_atoms,
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `qmin`.
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `qmin`.
            qmin=self.trainer_param.qmin,
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `qmax`.
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `qmax`.
            qmax=self.trainer_param.qmax,
        )

        q_network_target = q_network.get_target_network()

        return C51Trainer(
            q_network=q_network,
            q_network_target=q_network_target,
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )

    def build_serving_module(
        self,
        trainer_module: ReAgentLightningModule,
        normalization_data_map: Dict[str, NormalizationData],
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        assert isinstance(trainer_module, C51Trainer)
        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            trainer_module.q_network,
            normalization_data_map[NormalizationKey.STATE],
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )
