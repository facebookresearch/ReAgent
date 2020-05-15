#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Dict, List

import torch
from reagent.training.world_model.mdnrnn_trainer import MDNRNNTrainer
from reagent.types import FeatureData, MemoryNetworkInput


logger = logging.getLogger(__name__)


class LossEvaluator(object):
    """ Evaluate losses on data pages """

    def __init__(self, trainer: MDNRNNTrainer, state_dim: int) -> None:
        self.trainer = trainer
        self.state_dim = state_dim

    def evaluate(self, tdp: MemoryNetworkInput) -> Dict[str, float]:
        self.trainer.memory_network.mdnrnn.eval()
        losses = self.trainer.get_loss(tdp, state_dim=self.state_dim)
        detached_losses = {
            "loss": losses["loss"].cpu().detach().item(),
            "gmm": losses["gmm"].cpu().detach().item(),
            "bce": losses["bce"].cpu().detach().item(),
            "mse": losses["mse"].cpu().detach().item(),
        }
        del losses
        self.trainer.memory_network.mdnrnn.train()
        return detached_losses


class FeatureImportanceEvaluator(object):
    """ Evaluate feature importance weights on data pages """

    def __init__(
        self,
        trainer: MDNRNNTrainer,
        discrete_action: bool,
        state_feature_num: int,
        action_feature_num: int,
        sorted_action_feature_start_indices: List[int],
        sorted_state_feature_start_indices: List[int],
    ) -> None:
        """
        :param sorted_action_feature_start_indices: the starting index of each
            action feature in the action vector (need this because some features
            (e.g., one-hot encoding enum) may take multiple components)
        :param sorted_state_feature_start_indices: the starting index of each
            state feature in the state vector
        """
        self.trainer = trainer
        self.discrete_action = discrete_action
        self.state_feature_num = state_feature_num
        self.action_feature_num = action_feature_num
        self.sorted_action_feature_start_indices = sorted_action_feature_start_indices
        self.sorted_state_feature_start_indices = sorted_state_feature_start_indices

    def evaluate(self, batch: MemoryNetworkInput):
        """ Calculate feature importance: setting each state/action feature to
        the mean value and observe loss increase. """

        self.trainer.memory_network.mdnrnn.eval()
        state_features = batch.state.float_features
        action_features = batch.action
        seq_len, batch_size, state_dim = state_features.size()
        action_dim = action_features.size()[2]
        action_feature_num = self.action_feature_num
        state_feature_num = self.state_feature_num
        feature_importance = torch.zeros(action_feature_num + state_feature_num)

        orig_losses = self.trainer.get_loss(batch, state_dim=state_dim)
        orig_loss = orig_losses["loss"].cpu().detach().item()
        del orig_losses

        action_feature_boundaries = self.sorted_action_feature_start_indices + [
            action_dim
        ]
        state_feature_boundaries = self.sorted_state_feature_start_indices + [state_dim]

        for i in range(action_feature_num):
            action_features = batch.action.reshape(
                (batch_size * seq_len, action_dim)
            ).data.clone()

            # if actions are discrete, an action's feature importance is the loss
            # increase due to setting all actions to this action
            if self.discrete_action:
                assert action_dim == action_feature_num
                action_vec = torch.zeros(action_dim)
                action_vec[i] = 1
                action_features[:] = action_vec
            # if actions are continuous, an action's feature importance is the loss
            # increase due to masking this action feature to its mean value
            else:
                boundary_start, boundary_end = (
                    action_feature_boundaries[i],
                    action_feature_boundaries[i + 1],
                )
                action_features[
                    :, boundary_start:boundary_end
                ] = self.compute_median_feature_value(
                    action_features[:, boundary_start:boundary_end]
                )

            action_features = action_features.reshape((seq_len, batch_size, action_dim))
            new_batch = MemoryNetworkInput(
                state=batch.state,
                action=action_features,
                next_state=batch.next_state,
                reward=batch.reward,
                time_diff=torch.ones_like(batch.reward).float(),
                not_terminal=batch.not_terminal,
                step=None,
            )
            losses = self.trainer.get_loss(new_batch, state_dim=state_dim)
            feature_importance[i] = losses["loss"].cpu().detach().item() - orig_loss
            del losses

        for i in range(state_feature_num):
            state_features = batch.state.float_features.reshape(
                (batch_size * seq_len, state_dim)
            ).data.clone()
            boundary_start, boundary_end = (
                state_feature_boundaries[i],
                state_feature_boundaries[i + 1],
            )
            state_features[
                :, boundary_start:boundary_end
            ] = self.compute_median_feature_value(
                state_features[:, boundary_start:boundary_end]
            )
            state_features = state_features.reshape((seq_len, batch_size, state_dim))
            new_batch = MemoryNetworkInput(
                state=FeatureData(float_features=state_features),
                action=batch.action,
                next_state=batch.next_state,
                reward=batch.reward,
                time_diff=torch.ones_like(batch.reward).float(),
                not_terminal=batch.not_terminal,
                step=None,
            )
            losses = self.trainer.get_loss(new_batch, state_dim=state_dim)
            feature_importance[i + action_feature_num] = (
                losses["loss"].cpu().detach().item() - orig_loss
            )
            del losses

        self.trainer.memory_network.mdnrnn.train()
        logger.info(
            "**** Debug tool feature importance ****: {}".format(feature_importance)
        )
        return {"feature_loss_increase": feature_importance.numpy()}

    def compute_median_feature_value(self, features):
        # enum type
        if features.shape[1] > 1:
            feature_counts = torch.sum(features, dim=0)
            median_feature_counts = torch.median(feature_counts)
            # no similar method as numpy.where in torch
            for i in range(features.shape[1]):
                if feature_counts[i] == median_feature_counts:
                    break
            median_feature = torch.zeros(features.shape[1])
            median_feature[i] = 1
        # other types
        else:
            median_feature = features.mean(dim=0)
        return median_feature


class FeatureSensitivityEvaluator(object):
    """ Evaluate state feature sensitivity caused by varying actions """

    def __init__(
        self,
        trainer: MDNRNNTrainer,
        state_feature_num: int,
        sorted_state_feature_start_indices: List[int],
    ) -> None:
        self.trainer = trainer
        self.state_feature_num = state_feature_num
        self.sorted_state_feature_start_indices = sorted_state_feature_start_indices

    def evaluate(self, batch: MemoryNetworkInput):
        """ Calculate state feature sensitivity due to actions:
        randomly permutating actions and see how much the prediction of next
        state feature deviates. """
        assert isinstance(batch, MemoryNetworkInput)

        self.trainer.memory_network.mdnrnn.eval()

        seq_len, batch_size, state_dim = batch.next_state.float_features.size()
        state_feature_num = self.state_feature_num
        feature_sensitivity = torch.zeros(state_feature_num)

        # the input of world_model has seq-len as the first dimension
        mdnrnn_output = self.trainer.memory_network(
            batch.state, FeatureData(batch.action)
        )
        predicted_next_state_means = mdnrnn_output.mus

        shuffled_mdnrnn_output = self.trainer.memory_network(
            batch.state,
            # shuffle the actions
            FeatureData(batch.action[:, torch.randperm(batch_size), :]),
        )
        shuffled_predicted_next_state_means = shuffled_mdnrnn_output.mus

        assert (
            predicted_next_state_means.size()
            == shuffled_predicted_next_state_means.size()
            == (seq_len, batch_size, self.trainer.params.num_gaussians, state_dim)
        )

        state_feature_boundaries = self.sorted_state_feature_start_indices + [state_dim]
        for i in range(state_feature_num):
            boundary_start, boundary_end = (
                state_feature_boundaries[i],
                state_feature_boundaries[i + 1],
            )
            abs_diff = torch.mean(
                torch.sum(
                    torch.abs(
                        shuffled_predicted_next_state_means[
                            :, :, :, boundary_start:boundary_end
                        ]
                        - predicted_next_state_means[
                            :, :, :, boundary_start:boundary_end
                        ]
                    ),
                    dim=3,
                )
            )
            feature_sensitivity[i] = abs_diff.cpu().detach().item()

        self.trainer.memory_network.mdnrnn.train()
        logger.info(
            "**** Debug tool feature sensitivity ****: {}".format(feature_sensitivity)
        )
        return {"feature_sensitivity": feature_sensitivity.numpy()}
