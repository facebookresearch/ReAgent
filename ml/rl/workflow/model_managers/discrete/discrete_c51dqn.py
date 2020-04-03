#!/usr/bin/env python3

import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch  # @manual
from ml.rl import types as rlt
from ml.rl.net_builder.categorical_dqn.categorical import Categorical
from ml.rl.net_builder.categorical_dqn_net_builder import CategoricalDQNNetBuilder
from ml.rl.net_builder.unions import CategoricalDQNNetBuilder__Union
from ml.rl.parameters import param_hash
from ml.rl.training.c51_trainer import C51Trainer, C51TrainerParameters
from ml.rl.training.loss_reporter import NoOpLossReporter
from ml.rl.workflow.model_managers.discrete_dqn_base import DiscreteDQNBase


logger = logging.getLogger(__name__)


@dataclass
class DiscreteC51DQN(DiscreteDQNBase):
    __hash__ = param_hash

    trainer_param: C51TrainerParameters = field(default_factory=C51TrainerParameters)
    net_builder: CategoricalDQNNetBuilder__Union = field(
        default_factory=lambda: CategoricalDQNNetBuilder__Union(
            Categorical=Categorical()
        )
    )
    cpe_net_builder: CategoricalDQNNetBuilder__Union = field(
        default_factory=lambda: CategoricalDQNNetBuilder__Union(
            Categorical=Categorical()
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.rl_parameters = self.trainer_param.rl
        self.eval_parameters = self.trainer_param.evaluation
        self.action_names = self.trainer_param.actions
        assert len(self.action_names) > 1, "DiscreteC51DQN needs at least 2 actions"
        assert (
            self.trainer_param.minibatch_size % 8 == 0
        ), "The minibatch size must be divisible by 8 for performance reasons."

    def build_trainer(self) -> C51Trainer:
        net_builder = self.net_builder.value
        q_network = net_builder.build_q_network(
            self.state_normalization_parameters, len(self.action_names)
        )

        if self.use_gpu:
            q_network = q_network.cuda()

        q_network_target = q_network.get_target_network()

        self._q_network = q_network

        return C51Trainer(
            q_network,
            q_network_target,
            self.trainer_param,
            self.use_gpu,
            metrics_to_score=self.metrics_to_score,
            loss_reporter=NoOpLossReporter(),
        )

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
