#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.parameters import Seq2RewardTrainerParameters
from reagent.core.types import FeatureData
from reagent.models.fully_connected_network import FloatFeatureFullyConnected
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.utils import gen_permutations
from reagent.training.world_model.seq2reward_trainer import get_Q


logger = logging.getLogger(__name__)


class CompressModelTrainer(ReAgentLightningModule):
    """Trainer for fitting Seq2Reward planning outcomes to a neural network-based policy"""

    def __init__(
        self,
        compress_model_network: FloatFeatureFullyConnected,
        seq2reward_network: Seq2RewardNetwork,
        params: Seq2RewardTrainerParameters,
    ):
        super().__init__()
        self.compress_model_network = compress_model_network
        self.seq2reward_network = seq2reward_network
        self.params = params

        # permutations used to do planning
        self.all_permut = gen_permutations(
            params.multi_steps, len(self.params.action_names)
        )

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            {
                "optimizer": torch.optim.Adam(
                    self.compress_model_network.parameters(),
                    lr=self.params.compress_model_learning_rate,
                )
            }
        )
        return optimizers

    def train_step_gen(self, training_batch: rlt.MemoryNetworkInput, batch_idx: int):
        loss, accuracy = self.get_loss(training_batch)
        detached_loss = loss.cpu().detach().item()
        accuracy = accuracy.item()
        logger.info(
            f"Seq2Reward Compress trainer MSE/Accuracy: {detached_loss}, {accuracy}"
        )
        self.reporter.log(mse_loss=detached_loss, accuracy=accuracy)
        yield loss

    @staticmethod
    def extract_state_first_step(batch):
        return FeatureData(batch.state.float_features[0])

    # pyre-ignore inconsistent override because lightning doesn't use types
    def validation_step(self, batch: rlt.MemoryNetworkInput, batch_idx: int):
        mse, acc = self.get_loss(batch)
        detached_loss = mse.cpu().detach().item()
        acc = acc.item()

        state_first_step = CompressModelTrainer.extract_state_first_step(batch)
        # shape: batch_size, action_dim
        q_values_all_action_all_data = (
            self.compress_model_network(state_first_step).cpu().detach()
        )
        q_values = q_values_all_action_all_data.mean(0).tolist()

        action_distribution = torch.bincount(
            torch.argmax(q_values_all_action_all_data, dim=1),
            minlength=len(self.params.action_names),
        )
        # normalize
        action_distribution = (
            action_distribution.float() / torch.sum(action_distribution)
        ).tolist()

        self.reporter.log(
            eval_mse_loss=detached_loss,
            eval_accuracy=acc,
            eval_q_values=[q_values],
            eval_action_distribution=[action_distribution],
        )

        return (detached_loss, q_values, action_distribution, acc)

    def get_loss(self, batch: rlt.MemoryNetworkInput):
        state_first_step = CompressModelTrainer.extract_state_first_step(batch)
        # shape: batch_size, num_action
        compress_model_output = self.compress_model_network(state_first_step)

        target = get_Q(
            self.seq2reward_network,
            state_first_step.float_features,
            self.all_permut,
        )
        assert (
            compress_model_output.size() == target.size()
        ), f"{compress_model_output.size()}!={target.size()}"
        mse = F.mse_loss(compress_model_output, target)

        with torch.no_grad():
            target_action = torch.max(target, dim=1).indices
            model_action = torch.max(compress_model_output, dim=1).indices
            accuracy = torch.mean((target_action == model_action).float())

        return mse, accuracy

    def warm_start_components(self):
        logger.info("No warm start components yet...")
        components = []
        return components
