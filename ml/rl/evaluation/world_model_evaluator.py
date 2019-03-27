#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Dict

import torch
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
        self.trainer.mdnrnn.mdnrnn.train()
        return losses


class FeatureImportanceEvaluator(object):
    """ Evaluate feature importance weights on data pages """

    def __init__(self, trainer: MDNRNNTrainer) -> None:
        self.trainer = trainer

    def evaluate(self, tdp: TrainingBatch):
        """ Calculate feature importance: setting each state/action feature to
        the mean value and observe loss increase. """
        self.trainer.mdnrnn.mdnrnn.eval()

        state_features = tdp.training_input.state.float_features
        action_features = tdp.training_input.action.float_features
        batch_size, seq_len, state_dim = state_features.size()
        action_dim = action_features.size()[2]
        feature_importance = torch.zeros(action_dim + state_dim)
        orig_losses = self.trainer.get_loss(tdp, state_dim=state_dim, batch_first=True)
        orig_loss = orig_losses["loss"].item()

        for i in range(action_dim):
            action_features = tdp.training_input.action.float_features.reshape(
                (batch_size * seq_len, action_dim)
            ).data.clone()
            action_features[:, i] = action_features[:, i].mean()
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
            feature_importance[i] = losses["loss"].item() - orig_loss

        for i in range(state_dim):
            state_features = tdp.training_input.state.float_features.reshape(
                (batch_size * seq_len, state_dim)
            ).data.clone()
            state_features[:, i] = state_features[:, i].mean()
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
            feature_importance[i + action_dim] = losses["loss"].item() - orig_loss

        self.trainer.mdnrnn.mdnrnn.train()
        logger.info(
            "**** Debug tool feature importance ****: {}".format(feature_importance)
        )
        return {"feature_loss_increase": feature_importance}
