#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.parameters import Seq2RewardTrainerParameters
from reagent.core.torch_utils import get_device
from reagent.core.tracker import observable
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.training.loss_reporter import NoOpLossReporter
from reagent.training.trainer import Trainer
from reagent.training.utils import gen_permutations
from reagent.training.world_model.seq2reward_trainer import get_Q


logger = logging.getLogger(__name__)


@observable(mse_loss=torch.Tensor, accuracy=torch.Tensor)
class CompressModelTrainer(Trainer):
    """ Trainer for fitting Seq2Reward planning outcomes to a neural network-based policy """

    def __init__(
        self,
        compress_model_network: FullyConnectedNetwork,
        seq2reward_network: Seq2RewardNetwork,
        params: Seq2RewardTrainerParameters,
    ):
        self.compress_model_network = compress_model_network
        self.seq2reward_network = seq2reward_network
        self.params = params
        self.optimizer = torch.optim.Adam(
            self.compress_model_network.parameters(),
            lr=params.compress_model_learning_rate,
        )
        self.minibatch_size = self.params.compress_model_batch_size
        self.loss_reporter = NoOpLossReporter()

        # PageHandler must use this to activate evaluator:
        self.calc_cpe_in_training = True
        # permutations used to do planning
        device = get_device(self.compress_model_network)
        self.all_permut = gen_permutations(
            params.multi_steps, len(self.params.action_names)
        ).to(device)

    def train(self, training_batch: rlt.MemoryNetworkInput):
        self.optimizer.zero_grad()
        loss, accuracy = self.get_loss(training_batch)
        loss.backward()
        self.optimizer.step()
        detached_loss = loss.cpu().detach().item()
        accuracy = accuracy.item()
        logger.info(
            f"Seq2Reward Compress trainer MSE/Accuracy: {detached_loss}, {accuracy}"
        )
        # pyre-fixme[16]: `CompressModelTrainer` has no attribute
        #  `notify_observers`.
        self.notify_observers(mse_loss=detached_loss, accuracy=accuracy)
        return detached_loss, accuracy

    def get_loss(self, training_batch: rlt.MemoryNetworkInput):
        # shape: batch_size, num_action
        compress_model_output = self.compress_model_network(
            training_batch.state.float_features[0]
        )

        state_first_step = training_batch.state.float_features[0]
        target = get_Q(
            self.seq2reward_network,
            state_first_step,
            self.all_permut,
        )
        assert (
            compress_model_output.size() == target.size()
        ), f"{compress_model_output.size()}!={target.size()}"
        mse = F.mse_loss(compress_model_output, target)

        with torch.no_grad():
            # pyre-fixme[16]: `Tuple` has no attribute `indices`.
            target_action = torch.max(target, dim=1).indices
            model_action = torch.max(compress_model_output, dim=1).indices
            accuracy = torch.mean((target_action == model_action).float())

        return mse, accuracy

    def warm_start_components(self):
        logger.info("No warm start components yet...")
        components = []
        return components
