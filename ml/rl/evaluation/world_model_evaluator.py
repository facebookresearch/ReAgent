#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Dict

import torch
from ml.rl.training.training_data_page import TrainingDataPage
from ml.rl.training.world_model.mdnrnn_trainer import MDNRNNTrainer


class LossEvaluator(object):
    """ Evaluate losses on data pages """

    def __init__(self, trainer: MDNRNNTrainer, state_dim: int) -> None:
        self.trainer = trainer
        self.state_dim = state_dim

    def evaluate(self, tdp: TrainingDataPage) -> Dict:
        self.trainer.mdnrnn.mdnrnn.eval()
        losses = self.trainer.get_loss(tdp, state_dim=self.state_dim, batch_first=True)
        self.trainer.mdnrnn.mdnrnn.train()
        return losses


class FeatureImportanceEvaluator(object):
    """ Evaluate feature importance weights on data pages """

    def __init__(self, trainer: MDNRNNTrainer) -> None:
        self.trainer = trainer

    def evaluate(self, tdp: TrainingDataPage):
        """ Calculate feature importance: shuffle state and action feature
        values and observe loss increase. """
        self.trainer.mdnrnn.mdnrnn.eval()

        state_features = tdp.training_input.state.float_features
        action_features = tdp.training_input.action.float_features
        batch_size, seq_len, state_dim = state_features.size()
        action_dim = action_features.size()[2]
        feature_importance = torch.zeros(action_dim + state_dim)
        orig_losses = self.trainer.get_loss(tdp, state_dim=state_dim, batch_first=True)
        orig_loss = orig_losses["loss"].item()

        for i in range(action_dim):
            action_feature_backup = action_features[:, :, i].data.clone()
            action_features = action_features.reshape(
                (batch_size * seq_len, action_dim)
            )
            action_features[torch.arange(batch_size * seq_len), i] = action_features[
                torch.randperm(batch_size * seq_len), i
            ]
            action_features = action_features.reshape((batch_size, seq_len, action_dim))
            losses = self.trainer.get_loss(tdp, state_dim=state_dim, batch_first=True)
            feature_importance[i] = losses["loss"].item() - orig_loss
            action_features[:, :, i] = action_feature_backup

        for i in range(state_dim):
            state_feature_backup = state_features[:, :, i].data.clone()
            state_features = state_features.reshape((batch_size * seq_len, state_dim))
            state_features[torch.arange(batch_size * seq_len), i] = state_features[
                torch.randperm(batch_size * seq_len), i
            ]
            state_features = state_features.reshape((batch_size, seq_len, state_dim))
            losses = self.trainer.get_loss(tdp, state_dim=state_dim, batch_first=True)
            feature_importance[i + action_dim] = losses["loss"].item() - orig_loss
            state_features[:, :, i] = state_feature_backup

        self.trainer.mdnrnn.mdnrnn.train()
        return {"feature_loss_drop": feature_importance}
