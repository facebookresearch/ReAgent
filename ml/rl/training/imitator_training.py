#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.rl_trainer_pytorch import RLTrainer
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class ImitatorTrainer(RLTrainer):
    def __init__(
        self, imitator, parameters: DiscreteActionModelParameters, use_gpu=False
    ) -> None:
        self._set_optimizer(parameters.training.optimizer)
        self.minibatch_size = parameters.training.minibatch_size
        self.minibatches_per_step = parameters.training.minibatches_per_step or 1
        self.imitator = imitator
        self.imitator_optimizer = self.optimizer_func(
            imitator.parameters(),
            lr=parameters.training.learning_rate,
            weight_decay=parameters.training.l2_decay,
        )
        RLTrainer.__init__(self, parameters, use_gpu=use_gpu)

    def _imitator_accuracy(self, predictions, true_labels):
        match_tensor = predictions == true_labels
        matches = int(match_tensor.sum())
        return round(matches / len(predictions), 3)

    def train(self, training_batch, train=True):
        if isinstance(training_batch, TrainingDataPage):
            if self.maxq_learning:
                training_batch = training_batch.as_discrete_maxq_training_batch()
            else:
                training_batch = training_batch.as_discrete_sarsa_training_batch()
        learning_input = training_batch.training_input
        action_preds = self.imitator(learning_input.state.float_features)
        # Classification label is index of action with value 1
        pred_action_idxs = torch.max(action_preds, dim=1)[1]
        actual_action_idxs = torch.max(learning_input.action, dim=1)[1]

        if train:
            imitator_loss = torch.nn.CrossEntropyLoss()
            bcq_loss = imitator_loss(action_preds, actual_action_idxs)
            bcq_loss.backward()
            self._maybe_run_optimizer(
                self.imitator_optimizer, self.minibatches_per_step
            )

        return self._imitator_accuracy(pred_action_idxs, actual_action_idxs)


def get_valid_actions_from_imitator(imitator, input, drop_threshold):
    """Create mask for non-viable actions under the imitator."""
    if isinstance(imitator, torch.nn.Module):
        # pytorch model
        imitator_outputs = imitator(input)
        on_policy_action_probs = torch.nn.functional.softmax(imitator_outputs, dim=1)
    else:
        # sci-kit learn model
        on_policy_action_probs = torch.tensor(imitator(input.cpu()))

    filter_values = (
        on_policy_action_probs / on_policy_action_probs.max(keepdim=True, dim=1)[0]
    )
    return (filter_values >= drop_threshold).float()
