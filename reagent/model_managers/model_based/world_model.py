#!/usr/bin/env python3

import logging

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import MDNRNNTrainerParameters, param_hash
from reagent.model_managers.world_model_base import WorldModelBase
from reagent.models.world_model import MemoryNetwork
from reagent.preprocessing.normalization import get_num_output_features
from reagent.training.world_model.mdnrnn_trainer import MDNRNNTrainer


logger = logging.getLogger(__name__)


@dataclass
class WorldModel(WorldModelBase):
    __hash__ = param_hash

    trainer_param: MDNRNNTrainerParameters = field(
        default_factory=MDNRNNTrainerParameters
    )

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    # pyre-fixme[15]: `build_trainer` overrides method defined in `ModelManager`
    #  inconsistently.
    def build_trainer(self) -> MDNRNNTrainer:
        memory_network = MemoryNetwork(
            state_dim=get_num_output_features(
                self.state_normalization_data.dense_normalization_parameters
            ),
            action_dim=self.trainer_param.action_dim,
            num_hiddens=self.trainer_param.hidden_size,
            num_hidden_layers=self.trainer_param.num_hidden_layers,
            num_gaussians=self.trainer_param.num_gaussians,
        )
        if self.use_gpu:
            memory_network = memory_network.cuda()

        return MDNRNNTrainer(memory_network=memory_network, params=self.trainer_param)

    def build_serving_module(self) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        raise NotImplementedError()
