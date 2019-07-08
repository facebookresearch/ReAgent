#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Dict, List

import torch
from ml.rl.models.mdn_rnn import transpose
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

    def evaluate(self, tdp: TrainingBatch):
        """ Calculate feature importance: setting each state/action feature to
        the mean value and observe loss increase. """
        self.trainer.mdnrnn.mdnrnn.eval()

        state_features = tdp.training_input.state.float_features
        action_features = tdp.training_input.action.float_features  # type: ignore
        batch_size, seq_len, state_dim = state_features.size()  # type: ignore
        action_dim = action_features.size()[2]  # type: ignore
        action_feature_num = self.action_feature_num
        state_feature_num = self.state_feature_num
        feature_importance = torch.zeros(action_feature_num + state_feature_num)

        orig_losses = self.trainer.get_loss(tdp, state_dim=state_dim, batch_first=True)
        orig_loss = orig_losses["loss"].cpu().detach().item()
        del orig_losses

        action_feature_boundaries = self.sorted_action_feature_start_indices + [
            action_dim
        ]
        state_feature_boundaries = self.sorted_state_feature_start_indices + [state_dim]

        for i in range(action_feature_num):
            action_features = tdp.training_input.action.float_features.reshape(  # type: ignore
                (batch_size * seq_len, action_dim)
            ).data.clone()

            # if actions are discrete, an action's feature importance is the loss
            # increase due to setting all actions to this action
            if self.discrete_action:
                assert action_dim == action_feature_num
                action_vec = torch.zeros(action_dim)
                action_vec[i] = 1
                action_features[:] = action_vec  # type: ignore
            # if actions are continuous, an action's feature importance is the loss
            # increase due to masking this action feature to its mean value
            else:
                boundary_start, boundary_end = (
                    action_feature_boundaries[i],
                    action_feature_boundaries[i + 1],
                )
                action_features[  # type: ignore
                    :, boundary_start:boundary_end
                ] = self.compute_median_feature_value(  # type: ignore
                    action_features[:, boundary_start:boundary_end]  # type: ignore
                )

            action_features = action_features.reshape(  # type: ignore
                (batch_size, seq_len, action_dim)
            )  # type: ignore
            new_tdp = TrainingBatch(
                training_input=MemoryNetworkInput(  # type: ignore
                    state=tdp.training_input.state,
                    action=FeatureVector(  # type: ignore
                        float_features=action_features
                    ),  # type: ignore
                    next_state=tdp.training_input.next_state,
                    reward=tdp.training_input.reward,
                    time_diff=torch.ones_like(tdp.training_input.reward).float(),
                    not_terminal=tdp.training_input.not_terminal,  # type: ignore
                ),
                extras=ExtraData(),
            )
            losses = self.trainer.get_loss(
                new_tdp, state_dim=state_dim, batch_first=True
            )
            feature_importance[i] = losses["loss"].cpu().detach().item() - orig_loss
            del losses

        for i in range(state_feature_num):
            state_features = tdp.training_input.state.float_features.reshape(  # type: ignore
                (batch_size * seq_len, state_dim)
            ).data.clone()
            boundary_start, boundary_end = (
                state_feature_boundaries[i],
                state_feature_boundaries[i + 1],
            )
            state_features[  # type: ignore
                :, boundary_start:boundary_end
            ] = self.compute_median_feature_value(
                state_features[:, boundary_start:boundary_end]  # type: ignore
            )
            state_features = state_features.reshape(  # type: ignore
                (batch_size, seq_len, state_dim)
            )  # type: ignore
            new_tdp = TrainingBatch(
                training_input=MemoryNetworkInput(  # type: ignore
                    state=FeatureVector(float_features=state_features),  # type: ignore
                    action=tdp.training_input.action,  # type: ignore
                    next_state=tdp.training_input.next_state,
                    reward=tdp.training_input.reward,
                    time_diff=torch.ones_like(tdp.training_input.reward).float(),
                    not_terminal=tdp.training_input.not_terminal,  # type: ignore
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

    def evaluate(self, tdp: TrainingBatch):
        """ Calculate state feature sensitivity due to actions:
        randomly permutating actions and see how much the prediction of next
        state feature deviates. """
        self.trainer.mdnrnn.mdnrnn.eval()

        batch_size, seq_len, state_dim = (
            tdp.training_input.next_state.size()  # type: ignore
        )  # type: ignore
        state_feature_num = self.state_feature_num
        feature_sensitivity = torch.zeros(state_feature_num)

        mdnrnn_input = tdp.training_input
        state, action, next_state, reward, not_terminal = transpose(
            mdnrnn_input.state.float_features,
            mdnrnn_input.action.float_features,  # type: ignore
            mdnrnn_input.next_state,
            mdnrnn_input.reward,
            mdnrnn_input.not_terminal,  # type: ignore
        )
        mdnrnn_input = MemoryNetworkInput(  # type: ignore
            state=FeatureVector(float_features=state),
            action=FeatureVector(float_features=action),
            next_state=next_state,
            reward=reward,
            not_terminal=not_terminal,
            time_diff=torch.ones_like(reward),
        )
        # the input of mdnrnn has seq-len as the first dimension
        mdnrnn_output = self.trainer.mdnrnn(mdnrnn_input)
        predicted_next_state_means = mdnrnn_output.mus

        shuffled_mdnrnn_input = MemoryNetworkInput(  # type: ignore
            state=FeatureVector(float_features=state),
            # shuffle the actions
            action=FeatureVector(
                float_features=action[:, torch.randperm(batch_size), :]
            ),
            next_state=next_state,
            reward=reward,
            not_terminal=not_terminal,
            time_diff=torch.ones_like(reward).float(),
        )
        shuffled_mdnrnn_output = self.trainer.mdnrnn(shuffled_mdnrnn_input)
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

        self.trainer.mdnrnn.mdnrnn.train()
        logger.info(
            "**** Debug tool feature sensitivity ****: {}".format(feature_sensitivity)
        )
        return {"feature_sensitivity": feature_sensitivity.numpy()}
