#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Dict

import torch
from ml.rl.preprocessing.feature_extractor import WorldModelFeatureExtractor
from ml.rl.training.world_model.mdnrnn_trainer import MDNRNNTrainer
from ml.rl.types import ExtraData, FeatureVector, MemoryNetworkInput, TrainingBatch


logger = logging.getLogger(__name__)


class LossEvaluator(object):
    """ Evaluate losses on data pages """

    def __init__(self, trainer: MDNRNNTrainer, state_dim: int) -> None:
        self.trainer = trainer
        self.state_dim = state_dim

    def evaluate(self, tdp: TrainingBatch) -> Dict:
        self.trainer.mdnrnn.mdnrnn.eval()
        losses = self.trainer.get_loss(tdp, state_dim=self.state_dim, batch_first=True)
        detached_losses = {
            "loss": losses["loss"].cpu().detach().item(),
            "gmm": losses["gmm"].cpu().detach().item(),
            "bce": losses["bce"].cpu().detach().item(),
            "mse": losses["mse"].cpu().detach().item(),
        }
        del losses
        self.trainer.mdnrnn.mdnrnn.train()
        return detached_losses


class FeatureImportanceEvaluator(object):
    """ Evaluate feature importance weights on data pages """

    def __init__(
        self, trainer: MDNRNNTrainer, feature_extractor: WorldModelFeatureExtractor
    ) -> None:
        self.trainer = trainer
        self.feature_extractor = feature_extractor

    def evaluate(self, tdp: TrainingBatch):
        """ Calculate feature importance: setting each state/action feature to
        the mean value and observe loss increase. """
        self.trainer.mdnrnn.mdnrnn.eval()

        state_features = tdp.training_input.state.float_features
        action_features = tdp.training_input.action.float_features
        batch_size, seq_len, state_dim = state_features.size()
        action_dim = action_features.size()[2]
        action_feature_num = self.feature_extractor.action_feature_num
        state_feature_num = self.feature_extractor.state_feature_num
        feature_importance = torch.zeros(action_feature_num + state_feature_num)

        orig_losses = self.trainer.get_loss(tdp, state_dim=state_dim, batch_first=True)
        orig_loss = orig_losses["loss"].cpu().detach().item()
        del orig_losses

        action_feature_boundaries = (
            self.feature_extractor.sorted_action_feature_start_indices + [action_dim]
        )
        state_feature_boundaries = (
            self.feature_extractor.sorted_state_feature_start_indices + [state_dim]
        )

        for i in range(action_feature_num):
            action_features = tdp.training_input.action.float_features.reshape(
                (batch_size * seq_len, action_dim)
            ).data.clone()
            boundary_start, boundary_end = (
                action_feature_boundaries[i],
                action_feature_boundaries[i + 1],
            )
            action_features[:, boundary_start:boundary_end] = action_features[
                :, boundary_start:boundary_end
            ].mean(dim=0)
            action_features = action_features.reshape((batch_size, seq_len, action_dim))
            new_tdp = TrainingBatch(
                training_input=MemoryNetworkInput(
                    state=tdp.training_input.state,
                    action=FeatureVector(float_features=action_features),
                    next_state=tdp.training_input.next_state,
                    reward=tdp.training_input.reward,
                    not_terminal=tdp.training_input.not_terminal,
                ),
                extras=ExtraData(),
            )
            losses = self.trainer.get_loss(
                new_tdp, state_dim=state_dim, batch_first=True
            )
            feature_importance[i] = losses["loss"].cpu().detach().item() - orig_loss
            del losses

        for i in range(state_feature_num):
            state_features = tdp.training_input.state.float_features.reshape(
                (batch_size * seq_len, state_dim)
            ).data.clone()
            boundary_start, boundary_end = (
                state_feature_boundaries[i],
                state_feature_boundaries[i + 1],
            )
            state_features[:, boundary_start:boundary_end] = state_features[
                :, boundary_start:boundary_end
            ].mean(dim=0)
            state_features = state_features.reshape((batch_size, seq_len, state_dim))
            new_tdp = TrainingBatch(
                training_input=MemoryNetworkInput(
                    state=FeatureVector(float_features=state_features),
                    action=tdp.training_input.action,
                    next_state=tdp.training_input.next_state,
                    reward=tdp.training_input.reward,
                    not_terminal=tdp.training_input.not_terminal,
                ),
                extras=ExtraData(),
            )
            losses = self.trainer.get_loss(
                new_tdp, state_dim=state_dim, batch_first=True
            )
            feature_importance[i + action_feature_num] = (
                losses["loss"].cpu().detach().item() - orig_loss
            )
            del losses

        self.trainer.mdnrnn.mdnrnn.train()
        logger.info(
            "**** Debug tool feature importance ****: {}".format(feature_importance)
        )
        return {"feature_loss_increase": feature_importance.numpy()}
