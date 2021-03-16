#!/usr/bin/env python3

import logging

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import param_hash
from reagent.model_managers.discrete_dqn_base import DiscreteDQNBase
from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected
from reagent.net_builder.quantile_dqn.dueling_quantile import DuelingQuantile
from reagent.net_builder.unions import (
    DiscreteDQNNetBuilder__Union,
    QRDQNNetBuilder__Union,
)
from reagent.training import QRDQNTrainer, QRDQNTrainerParameters


logger = logging.getLogger(__name__)


@dataclass
class DiscreteQRDQN(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: QRDQNTrainerParameters = field(
        default_factory=QRDQNTrainerParameters
    )
    net_builder: QRDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `DuelingQuantile`.
        default_factory=lambda: QRDQNNetBuilder__Union(
            DuelingQuantile=DuelingQuantile()
        )
    )
    cpe_net_builder: DiscreteDQNNetBuilder__Union = field(
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`
        default_factory=lambda: DiscreteDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        self.rl_parameters = self.trainer_param.rl
        self.action_names = self.trainer_param.actions
        assert len(self.action_names) > 1, "DiscreteQRDQNModel needs at least 2 actions"
        assert (
            self.trainer_param.minibatch_size % 8 == 0
        ), "The minibatch size must be divisible by 8 for performance reasons."

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(self) -> QRDQNTrainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            self.state_normalization_data,
            len(self.action_names),
            # pyre-fixme[16]: `QRDQNTrainerParameters` has no attribute `num_atoms`.
            num_atoms=self.trainer_param.num_atoms,
        )

        q_network_target = q_network.get_target_network()

        reward_network, q_network_cpe, q_network_cpe_target = None, None, None
        if self.eval_parameters.calc_cpe_in_training:
            # Metrics + reward
            num_output_nodes = (len(self.metrics_to_score) + 1) * len(
                # pyre-fixme[16]: `QRDQNTrainerParameters` has no attribute `actions`.
                self.trainer_param.actions
            )

            cpe_net_builder = self.cpe_net_builder.value
            reward_network = cpe_net_builder.build_q_network(
                self.state_feature_config,
                self.state_normalization_data,
                num_output_nodes,
            )
            q_network_cpe = cpe_net_builder.build_q_network(
                self.state_feature_config,
                self.state_normalization_data,
                num_output_nodes,
            )

            q_network_cpe_target = q_network_cpe.get_target_network()

        # pyre-fixme[16]: `DiscreteQRDQN` has no attribute `_q_network`.
        self._q_network = q_network
        trainer = QRDQNTrainer(
            q_network=q_network,
            q_network_target=q_network_target,
            reward_network=reward_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=self.metrics_to_score,
            evaluation=self.eval_parameters,
            # pyre-fixme[16]: `QRDQNTrainerParameters` has no attribute `asdict`.
            **self.trainer_param.asdict(),
        )
        return trainer

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
