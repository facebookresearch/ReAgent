#!/usr/bin/env python3

import logging

import torch  # @manual
from reagent.core.dataclasses import dataclass, field
from reagent.models.world_model import MemoryNetwork
from reagent.parameters import param_hash
from reagent.training.world_model.mdnrnn_trainer import (
    MDNRNNTrainer,
    MDNRNNTrainerParameters,
)
from reagent.workflow.model_managers.world_model_base import WorldModelBase


logger = logging.getLogger(__name__)


@dataclass
class WorldModel(WorldModelBase):
    __hash__ = param_hash

    state_dim: int
    action_dim: int
    use_gpu: bool
    trainer_param: MDNRNNTrainerParameters = field(
        default_factory=MDNRNNTrainerParameters
    )

    def build_trainer(self) -> MDNRNNTrainer:
        mdnrnn_net = MemoryNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_hiddens=self.trainer_param.hidden_size,
            num_hidden_layers=self.trainer_param.num_hidden_layers,
            num_gaussians=self.trainer_param.num_gaussians,
        )
        if self.use_gpu:
            mdnrnn_net = mdnrnn_net.cuda()

        return MDNRNNTrainer(mdnrnn_network=mdnrnn_net, params=self.trainer_param)

    def build_serving_module(self) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        raise NotImplementedError()
