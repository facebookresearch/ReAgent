#!/usr/bin/env python3

import copy
import logging
from typing import List, Optional

import torch  # @manual
from ml.rl import types as rlt
from ml.rl.core.dataclasses import dataclass, field
from ml.rl.net_builder.discrete_dqn.dueling import Dueling
from ml.rl.net_builder.discrete_dqn.fully_connected import FullyConnected
from ml.rl.net_builder.discrete_dqn_net_builder import DiscreteDQNNetBuilder
from ml.rl.net_builder.unions import DiscreteDQNNetBuilder__Union
from ml.rl.parameters import param_hash
from ml.rl.training.dqn_trainer import DQNTrainer, DQNTrainerParameters
from ml.rl.training.loss_reporter import NoOpLossReporter
from ml.rl.workflow.model_managers.discrete_dqn_base import DiscreteDQNBase


logger = logging.getLogger(__name__)


@dataclass
class DiscreteDQN(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: DQNTrainerParameters = field(default_factory=DQNTrainerParameters)
    net_builder: DiscreteDQNNetBuilder__Union = field(
        default_factory=lambda: DiscreteDQNNetBuilder__Union(Dueling=Dueling())
    )
    cpe_net_builder: DiscreteDQNNetBuilder__Union = field(
        default_factory=lambda: DiscreteDQNNetBuilder__Union(
            FullyConnected=FullyConnected()
        )
    )

    def __post_init_post_parse__(self,):
        super().__post_init_post_parse__()
        self.rl_parameters = self.trainer_param.rl
        self.eval_parameters = self.trainer_param.evaluation
        self.action_names = self.trainer_param.actions
        assert len(self.action_names) > 1, "DiscreteDQNModel needs at least 2 actions"
        assert (
            self.trainer_param.minibatch_size % 8 == 0
        ), "The minibatch size must be divisible by 8 for performance reasons."

    def build_trainer(self) -> DQNTrainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            self.state_feature_config,
            self.state_normalization_parameters,
            len(self.action_names),
        )

        if self.use_gpu:
            q_network = q_network.cuda()

        q_network_target = q_network.get_target_network()

        reward_network, q_network_cpe, q_network_cpe_target = None, None, None
        if self.trainer_param.evaluation.calc_cpe_in_training:
            # Metrics + reward
            num_output_nodes = (len(self.metrics_to_score) + 1) * len(
                self.trainer_param.actions
            )

            cpe_net_builder = self.cpe_net_builder.value
            reward_network = cpe_net_builder.build_q_network(
                self.state_feature_config,
                self.state_normalization_parameters,
                num_output_nodes,
            )
            q_network_cpe = cpe_net_builder.build_q_network(
                self.state_feature_config,
                self.state_normalization_parameters,
                num_output_nodes,
            )

            if self.use_gpu:
                reward_network.cuda()
                q_network_cpe.cuda()

            q_network_cpe_target = q_network_cpe.get_target_network()

        self._q_network = q_network
        trainer = DQNTrainer(
            q_network,
            q_network_target,
            reward_network,
            self.trainer_param,
            self.use_gpu,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=self.metrics_to_score,
            loss_reporter=NoOpLossReporter(),
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
            self.state_normalization_parameters,
            action_names=self.action_names,
            state_feature_config=self.state_feature_config,
        )
