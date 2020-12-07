#!/usr/bin/env python3

import logging

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.net_builder.categorical_dqn.categorical import Categorical
from reagent.net_builder.unions import CategoricalDQNNetBuilder__Union
from reagent.parameters import param_hash
from reagent.training import C51Trainer, C51TrainerParameters
from reagent.workflow.model_managers.discrete_dqn_base import DiscreteDQNBase


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
        self.rl_parameters = self.trainer_param.rl
        self.action_names = self.trainer_param.actions
        assert len(self.action_names) > 1, "DiscreteC51DQN needs at least 2 actions"
        assert (
            self.trainer_param.minibatch_size % 8 == 0
        ), "The minibatch size must be divisible by 8 for performance reasons."

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(self) -> C51Trainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            state_normalization_data=self.state_normalization_data,
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

        # pyre-fixme[16]: `DiscreteC51DQN` has no attribute `_q_network`.
        # pyre-fixme[16]: `DiscreteC51DQN` has no attribute `_q_network`.
        self._q_network = q_network

        return C51Trainer(
            q_network=q_network,
            q_network_target=q_network_target,
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `asdict`.
            # pyre-fixme[16]: `C51TrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )

    def build_serving_module(self) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        assert self._q_network is not None, "_q_network was not initialized"
        net_builder = self.net_builder.value
        return net_builder.build_serving_module(
            self._q_network,
            self.state_normalization_data,
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )
