#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import NamedTuple, Optional

import numpy as np
import torch
from ml.rl.caffe_utils import masked_softmax
from ml.rl.training.rl_trainer_pytorch import RLTrainer
from ml.rl.training.training_data_page import TrainingDataPage
from ml.rl.types import TrainingBatch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EvaluationDataPage(NamedTuple):
    mdp_id: np.ndarray
    sequence_number: torch.Tensor
    logged_propensities: torch.Tensor
    logged_rewards: torch.Tensor
    action_mask: torch.Tensor
    model_propensities: torch.Tensor
    model_rewards: torch.Tensor
    model_rewards_for_logged_action: torch.Tensor
    model_values: torch.Tensor
    model_values_for_logged_action: torch.Tensor
    possible_actions_mask: torch.Tensor
    logged_values: Optional[torch.Tensor] = None
    logged_metrics: Optional[torch.Tensor] = None
    logged_metrics_values: Optional[torch.Tensor] = None
    model_metrics: Optional[torch.Tensor] = None
    model_metrics_for_logged_action: Optional[torch.Tensor] = None
    model_metrics_values: Optional[torch.Tensor] = None
    model_metrics_values_for_logged_action: Optional[torch.Tensor] = None
    possible_actions_state_concat: Optional[torch.Tensor] = None

    @classmethod
    def create_from_tdp(cls, tdp: TrainingDataPage, trainer: RLTrainer):
        return EvaluationDataPage.create_from_tensors(
            trainer,
            tdp.mdp_ids,
            tdp.sequence_numbers,
            tdp.states,
            tdp.actions,
            tdp.propensities,
            tdp.rewards,
            tdp.possible_actions_state_concat,
            tdp.possible_actions_mask,
            tdp.metrics,
        )

    @classmethod
    def create_from_training_batch(cls, tdp: TrainingBatch, trainer: RLTrainer):
        return EvaluationDataPage.create_from_tensors(
            trainer,
            tdp.mdp_ids,
            tdp.sequence_numbers,
            tdp.states,
            tdp.actions,
            tdp.propensities,
            tdp.rewards,
            tdp.possible_actions_state_concat,
            tdp.possible_actions_mask,
            tdp.metrics,
        )

    @classmethod
    def create_from_tensors(
        cls,
        trainer: RLTrainer,
        mdp_ids: np.ndarray,
        sequence_numbers: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        propensities: torch.Tensor,
        rewards: torch.Tensor,
        possible_actions_state_concat: Optional[torch.Tensor],
        possible_actions_mask: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            # Switch to evaluation mode for the network
            old_q_train_state = trainer.q_network.training
            old_reward_train_state = trainer.reward_network.training
            trainer.q_network.train(False)
            trainer.reward_network.train(False)

            if possible_actions_state_concat is not None:
                state_action_pairs = torch.cat((states, actions), dim=1)

                # Parametric actions
                rewards = rewards
                model_values = trainer.q_network(possible_actions_state_concat)
                assert (
                    model_values.shape[0] * model_values.shape[1]
                    == possible_actions_mask.shape[0] * possible_actions_mask.shape[1]
                ), (
                    "Invalid shapes: "
                    + str(model_values.shape)
                    + " != "
                    + str(possible_actions_mask.shape)
                )
                model_values = model_values.reshape(possible_actions_mask.shape)

                model_rewards = trainer.reward_network(possible_actions_state_concat)
                assert (
                    model_rewards.shape[0] * model_rewards.shape[1]
                    == possible_actions_mask.shape[0] * possible_actions_mask.shape[1]
                ), (
                    "Invalid shapes: "
                    + str(model_rewards.shape)
                    + " != "
                    + str(possible_actions_mask.shape)
                )
                model_rewards = model_rewards.reshape(possible_actions_mask.shape)

                model_values_for_logged_action = trainer.q_network(state_action_pairs)
                model_rewards_for_logged_action = trainer.reward_network(
                    state_action_pairs
                )

                action_mask = (
                    torch.abs(model_values - model_values_for_logged_action) < 1e-3
                ).float()

                model_metrics = None
                model_metrics_for_logged_action = None
                model_metrics_values = None
                model_metrics_values_for_logged_action = None
            else:
                action_mask = actions.float()

                # Switch to evaluation mode for the network
                old_q_cpe_train_state = trainer.q_network_cpe.training
                trainer.q_network_cpe.train(False)

                # Discrete actions
                rewards = trainer.boost_rewards(rewards, actions)
                model_values = trainer.get_detached_q_values(states)[0]
                assert model_values.shape == actions.shape, (
                    "Invalid shape: "
                    + str(model_values.shape)
                    + " != "
                    + str(actions.shape)
                )
                assert model_values.shape == possible_actions_mask.shape, (
                    "Invalid shape: "
                    + str(model_values.shape)
                    + " != "
                    + str(possible_actions_mask.shape)
                )
                model_values_for_logged_action = torch.sum(
                    model_values * action_mask, dim=1, keepdim=True
                )

                rewards_and_metric_rewards = trainer.reward_network(states)

                num_actions = trainer.num_actions

                model_rewards = rewards_and_metric_rewards[:, 0:num_actions]
                assert model_rewards.shape == actions.shape, (
                    "Invalid shape: "
                    + str(model_rewards.shape)
                    + " != "
                    + str(actions.shape)
                )
                model_rewards_for_logged_action = torch.sum(
                    model_rewards * action_mask, dim=1, keepdim=True
                )

                model_metrics = rewards_and_metric_rewards[:, num_actions:]

                assert model_metrics.shape[1] % num_actions == 0, (
                    "Invalid metrics shape: "
                    + str(model_metrics.shape)
                    + " "
                    + str(num_actions)
                )
                num_metrics = model_metrics.shape[1] // num_actions

                if num_metrics == 0:
                    model_metrics_values = None
                    model_metrics_for_logged_action = None
                    model_metrics_values_for_logged_action = None
                else:
                    model_metrics_values = trainer.q_network_cpe(states)[
                        :, num_actions:
                    ]
                    assert model_metrics_values.shape[1] == num_actions * num_metrics, (
                        "Invalid shape: "
                        + str(model_metrics_values.shape[1])
                        + " != "
                        + str(actions.shape[1] * num_metrics)
                    )

                    model_metrics_for_logged_action_list = []
                    model_metrics_values_for_logged_action_list = []
                    for metric_index in range(num_metrics):
                        metric_start = metric_index * num_actions
                        metric_end = (metric_index + 1) * num_actions
                        model_metrics_for_logged_action_list.append(
                            torch.sum(
                                model_metrics[:, metric_start:metric_end] * action_mask,
                                dim=1,
                                keepdim=True,
                            )
                        )

                        model_metrics_values_for_logged_action_list.append(
                            torch.sum(
                                model_metrics_values[:, metric_start:metric_end]
                                * action_mask,
                                dim=1,
                                keepdim=True,
                            )
                        )
                    model_metrics_for_logged_action = torch.cat(
                        model_metrics_for_logged_action_list, dim=1
                    )
                    model_metrics_values_for_logged_action = torch.cat(
                        model_metrics_values_for_logged_action_list, dim=1
                    )

                # Switch back to the old mode
                trainer.q_network_cpe.train(old_q_cpe_train_state)

            # Switch back to the old mode
            trainer.q_network.train(old_q_train_state)
            trainer.reward_network.train(old_reward_train_state)

            return cls(
                mdp_id=mdp_ids,
                sequence_number=sequence_numbers,
                logged_propensities=propensities,
                logged_rewards=rewards,
                action_mask=action_mask,
                model_rewards=model_rewards,
                model_rewards_for_logged_action=model_rewards_for_logged_action,
                model_values=model_values,
                model_values_for_logged_action=model_values_for_logged_action,
                model_metrics_values=model_metrics_values,
                model_metrics_values_for_logged_action=model_metrics_values_for_logged_action,
                model_propensities=masked_softmax(
                    model_values, possible_actions_mask, trainer.rl_temperature
                ),
                logged_metrics=metrics,
                model_metrics=model_metrics,
                model_metrics_for_logged_action=model_metrics_for_logged_action,
                # Will compute later
                logged_values=None,
                logged_metrics_values=None,
                possible_actions_state_concat=possible_actions_state_concat,
                possible_actions_mask=possible_actions_mask,
            )

    def append(self, edp):
        new_edp = {}
        for x in EvaluationDataPage._fields:
            t = getattr(self, x)
            other_t = getattr(edp, x)
            assert int(t is not None) + int(other_t is not None) != 1, (
                "Tried to append when a tensor existed in one training page but not the other: "
                + x
            )
            if other_t is not None:
                if isinstance(t, torch.Tensor):
                    new_edp[x] = torch.cat((t, other_t), dim=0)
                elif isinstance(t, np.ndarray):
                    new_edp[x] = np.concatenate((t, other_t), axis=0)
                else:
                    raise Exception("Invalid type in training data page")
            else:
                new_edp[x] = None
        return EvaluationDataPage(**new_edp)

    def compute_values(self, gamma: float):
        logged_values = EvaluationDataPage.compute_values_for_mdps(
            self.logged_rewards, self.mdp_id, self.sequence_number, gamma
        )
        if self.logged_metrics is not None:
            logged_metrics_values = EvaluationDataPage.compute_values_for_mdps(
                self.logged_metrics, self.mdp_id, self.sequence_number, gamma
            )
        else:
            logged_metrics_values = None
        return self._replace(
            logged_values=logged_values, logged_metrics_values=logged_metrics_values
        )

    @staticmethod
    def compute_values_for_mdps(
        rewards: torch.Tensor,
        mdp_ids: np.ndarray,
        sequence_numbers: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        values = rewards.clone()

        for x in range(len(values) - 2, -1, -1):
            if mdp_ids[x] != mdp_ids[x + 1]:
                # Value = reward for new mdp_id
                continue
            values[x, 0] += values[x + 1, 0] * math.pow(
                gamma, float(sequence_numbers[x + 1, 0] - sequence_numbers[x, 0])
            )

        return values

    def validate(self):
        assert len(self.logged_propensities.shape) == 2
        assert len(self.logged_rewards.shape) == 2
        assert len(self.logged_values.shape) == 2
        assert len(self.model_propensities.shape) == 2
        assert len(self.model_rewards.shape) == 2
        assert len(self.model_values.shape) == 2

        assert self.logged_propensities.shape[1] == 1
        assert self.logged_rewards.shape[1] == 1
        assert self.logged_values.shape[1] == 1
        num_actions = self.model_propensities.shape[1]
        assert self.model_rewards.shape[1] == num_actions
        assert self.model_values.shape[1] == num_actions

        assert self.action_mask.shape == self.model_propensities.shape

        if self.logged_metrics is not None:
            assert len(self.logged_metrics.shape) == 2
            assert len(self.logged_metrics_values.shape) == 2
            assert len(self.model_metrics.shape) == 2
            assert len(self.model_metrics_values.shape) == 2

            num_metrics = self.logged_metrics.shape[1]
            assert self.logged_metrics_values.shape[1] == num_metrics, (
                "Invalid shape: "
                + str(self.logged_metrics_values.shape)
                + " != "
                + str(num_metrics)
            )
            assert self.model_metrics.shape[1] == num_metrics * num_actions, (
                "Invalid shape: "
                + str(self.model_metrics.shape)
                + " != "
                + str(num_metrics * num_actions)
            )
            assert self.model_metrics_values.shape[1] == num_metrics * num_actions

        minibatch_size = self.logged_propensities.shape[0]
        assert minibatch_size == self.logged_rewards.shape[0]
        assert minibatch_size == self.logged_values.shape[0]
        assert minibatch_size == self.model_propensities.shape[0]
        assert minibatch_size == self.model_rewards.shape[0]
        assert minibatch_size == self.model_values.shape[0]
        if self.logged_metrics is not None:
            assert minibatch_size == self.logged_metrics.shape[0]
            assert minibatch_size == self.logged_metrics_values.shape[0]
            assert minibatch_size == self.model_metrics.shape[0]
            assert minibatch_size == self.model_metrics_values.shape[0]

    def set_metric_as_reward(self, i: int, num_actions: int):
        assert self.logged_metrics is not None, "metrics must not be none"
        assert self.logged_metrics_values is not None, "metrics must not be none"
        assert self.model_metrics is not None, "metrics must not be none"
        assert self.model_metrics_values is not None, "metrics must not be none"

        return self._replace(
            logged_rewards=self.logged_metrics[:, i : i + 1],
            logged_values=self.logged_metrics_values[:, i : i + 1],
            model_rewards=self.model_metrics[
                :, i * num_actions : (i + 1) * num_actions
            ],
            model_values=self.model_metrics_values[
                :, i * num_actions : (i + 1) * num_actions
            ],
            logged_metrics=None,
            logged_metrics_values=None,
            model_metrics=None,
            model_metrics_values=None,
        )
