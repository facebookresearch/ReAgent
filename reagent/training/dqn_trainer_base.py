#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional

import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.parameters import EvaluationParameters, RLParameters
from reagent.core.torch_utils import masked_softmax
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.evaluator import Evaluator
from reagent.optimizer import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.rl_trainer_pytorch import RLTrainerMixin

logger = logging.getLogger(__name__)


class DQNTrainerMixin:
    # Q-value for action that is not possible. Guaranteed to be worse than any
    # legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9

    def get_max_q_values(self, q_values, possible_actions_mask):
        return self.get_max_q_values_with_target(
            q_values, q_values, possible_actions_mask
        )

    def get_max_q_values_with_target(
        self, q_values, q_values_target, possible_actions_mask
    ):
        """
        Used in Q-learning update.

        :param q_values: PyTorch tensor with shape (batch_size, state_dim). Each row
            contains the list of Q-values for each possible action in this state.

        :param q_values_target: PyTorch tensor with shape (batch_size, state_dim). Each row
            contains the list of Q-values from the target network
            for each possible action in this state.

        :param possible_actions_mask: PyTorch tensor with shape (batch_size, action_dim).
            possible_actions[i][j] = 1 iff the agent can take action j from
            state i.

        Returns a tensor of maximum Q-values for every state in the batch
            and also the index of the corresponding action (which is used in
            evaluation_data_page.py, in create_from_tensors_dqn()).

        """

        # The parametric DQN can create flattened q values so we reshape here.
        q_values = q_values.reshape(possible_actions_mask.shape)
        q_values_target = q_values_target.reshape(possible_actions_mask.shape)
        # Set q-values of impossible actions to a very large negative number.
        inverse_pna = 1 - possible_actions_mask
        impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
        q_values = q_values + impossible_action_penalty
        q_values_target = q_values_target + impossible_action_penalty

        if self.double_q_learning:
            # Use indices of the max q_values from the online network to select q-values
            # from the target network. This prevents overestimation of q-values.
            # The torch.gather function selects the entry from each row that corresponds
            # to the max_index in that row.
            max_q_values, max_indicies = torch.max(q_values, dim=1, keepdim=True)
            max_q_values_target = torch.gather(q_values_target, 1, max_indicies)
        else:
            max_q_values_target, max_indicies = torch.max(
                q_values_target, dim=1, keepdim=True
            )

        return max_q_values_target, max_indicies


class DQNTrainerBaseLightning(DQNTrainerMixin, RLTrainerMixin, ReAgentLightningModule):
    def __init__(
        self,
        rl_parameters: RLParameters,
        metrics_to_score=None,
        actions: Optional[List[str]] = None,
        evaluation_parameters: Optional[EvaluationParameters] = None,
    ):
        super().__init__()
        self.rl_parameters = rl_parameters
        self.time_diff_unit_length = rl_parameters.time_diff_unit_length
        self.tensorboard_logging_freq = rl_parameters.tensorboard_logging_freq
        self.calc_cpe_in_training = (
            evaluation_parameters and evaluation_parameters.calc_cpe_in_training
        )
        assert actions is not None
        self._actions: List[str] = actions

        if rl_parameters.q_network_loss == "mse":
            self.q_network_loss = F.mse_loss
        elif rl_parameters.q_network_loss == "huber":
            self.q_network_loss = F.smooth_l1_loss
        else:
            raise Exception(
                "Q-Network loss type {} not valid loss.".format(
                    rl_parameters.q_network_loss
                )
            )

        if metrics_to_score:
            self.metrics_to_score = metrics_to_score + ["reward"]
        else:
            self.metrics_to_score = ["reward"]

    def _check_input(self, training_batch: rlt.DiscreteDqnInput):
        assert isinstance(training_batch, rlt.DiscreteDqnInput)
        assert training_batch.not_terminal.dim() == training_batch.reward.dim() == 2
        assert (
            training_batch.not_terminal.shape[1] == training_batch.reward.shape[1] == 1
        )
        assert training_batch.action.dim() == training_batch.next_action.dim() == 2
        assert (
            training_batch.action.shape[1]
            == training_batch.next_action.shape[1]
            == self.num_actions
        )

    @property
    def num_actions(self) -> int:
        assert self._actions is not None, "Not a discrete action DQN"
        return len(self._actions)

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def boost_rewards(
        self, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            actions.float() * self.reward_boosts,
            dim=1,
            keepdim=True,
        )
        return rewards + reward_boosts

    def _initialize_cpe(
        self,
        reward_network,
        q_network_cpe,
        q_network_cpe_target,
        optimizer: Optimizer__Union,
    ) -> None:
        if not self.calc_cpe_in_training:
            # pyre-fixme[16]: `DQNTrainerBase` has no attribute `reward_network`.
            self.reward_network = None
            return

        assert reward_network is not None, "reward_network is required for CPE"
        self.reward_network = reward_network
        # pyre-fixme[16]: `DQNTrainerBase` has no attribute `reward_network_optimizer`.
        self.reward_network_optimizer = optimizer
        assert (
            q_network_cpe is not None and q_network_cpe_target is not None
        ), "q_network_cpe and q_network_cpe_target are required for CPE"
        # pyre-fixme[16]: `DQNTrainerBase` has no attribute `q_network_cpe`.
        self.q_network_cpe = q_network_cpe
        # pyre-fixme[16]: `DQNTrainerBase` has no attribute `q_network_cpe_target`.
        self.q_network_cpe_target = q_network_cpe_target
        # pyre-fixme[16]: `DQNTrainerBase` has no attribute `q_network_cpe_optimizer`.
        self.q_network_cpe_optimizer = optimizer
        num_output_nodes = len(self.metrics_to_score) * self.num_actions
        reward_idx_offsets = torch.arange(
            0,
            num_output_nodes,
            self.num_actions,
            dtype=torch.long,
        )
        self.register_buffer("reward_idx_offsets", reward_idx_offsets)

        reward_stripped_metrics_to_score = (
            self.metrics_to_score[:-1] if len(self.metrics_to_score) > 1 else None
        )
        # pyre-fixme[16]: `DQNTrainerBase` has no attribute `evaluator`.
        self.evaluator = Evaluator(
            self._actions,
            self.rl_parameters.gamma,
            self,
            metrics_to_score=reward_stripped_metrics_to_score,
        )

    def _calculate_cpes(
        self,
        training_batch,
        states,
        next_states,
        all_action_scores,
        all_next_action_scores,
        logged_action_idxs,
        discount_tensor,
        not_done_mask,
    ):
        if not self.calc_cpe_in_training:
            return
        if training_batch.extras.metrics is None:
            metrics_reward_concat_real_vals = training_batch.reward
        else:
            metrics_reward_concat_real_vals = torch.cat(
                (training_batch.reward, training_batch.extras.metrics), dim=1
            )

        model_propensities_next_states = masked_softmax(
            all_next_action_scores,
            training_batch.possible_next_actions_mask
            if self.maxq_learning
            else training_batch.next_action,
            self.rl_temperature,
        )

        ######### Train separate reward network for CPE evaluation #############
        reward_estimates = self.reward_network(states)
        reward_estimates_for_logged_actions = reward_estimates.gather(
            1, self.reward_idx_offsets + logged_action_idxs
        )
        reward_loss = F.mse_loss(
            reward_estimates_for_logged_actions, metrics_reward_concat_real_vals
        )
        yield reward_loss

        ######### Train separate q-network for CPE evaluation #############
        metric_q_values = self.q_network_cpe(states).gather(
            1, self.reward_idx_offsets + logged_action_idxs
        )
        all_metrics_target_q_values = torch.chunk(
            self.q_network_cpe_target(next_states).detach(),
            len(self.metrics_to_score),
            dim=1,
        )
        target_metric_q_values = []
        for i, per_metric_target_q_values in enumerate(all_metrics_target_q_values):
            per_metric_next_q_values = torch.sum(
                per_metric_target_q_values * model_propensities_next_states,
                1,
                keepdim=True,
            )
            per_metric_next_q_values = per_metric_next_q_values * not_done_mask
            per_metric_target_q_values = metrics_reward_concat_real_vals[
                :, i : i + 1
            ] + (discount_tensor * per_metric_next_q_values)
            target_metric_q_values.append(per_metric_target_q_values)

        target_metric_q_values = torch.cat(target_metric_q_values, dim=1)
        metric_q_value_loss = self.q_network_loss(
            metric_q_values, target_metric_q_values
        )

        # The model_propensities computed below are not used right now. The CPE graphs in the Outputs
        # tab use model_propensities computed in the function create_from_tensors_dqn() in evaluation_data_page.py,
        # which is called on the eval_table_sample in the gather_eval_data() function below.
        model_propensities = masked_softmax(
            all_action_scores,
            training_batch.possible_actions_mask
            if self.maxq_learning
            else training_batch.action,
            self.rl_temperature,
        )
        # Extract rewards predicted by the reward_network. The other columns will
        # give predicted values for other metrics, if such were specified.
        model_rewards = reward_estimates[
            :,
            torch.arange(
                self.reward_idx_offsets[0],
                self.reward_idx_offsets[0] + self.num_actions,
            ),
        ]

        self.reporter.log(
            reward_loss=reward_loss,
            model_propensities=model_propensities,
            model_rewards=model_rewards,
        )

        yield metric_q_value_loss

    def gather_eval_data(self, validation_step_outputs):
        was_on_gpu = self.on_gpu
        self.cpu()
        eval_data = None
        for edp in validation_step_outputs:
            if eval_data is None:
                eval_data = edp
            else:
                eval_data = eval_data.append(edp)
        if eval_data and eval_data.mdp_id is not None:
            eval_data = eval_data.sort()
            eval_data = eval_data.compute_values(self.gamma)
            eval_data.validate()
        if was_on_gpu:
            self.cuda()
        return eval_data

    def validation_step(self, batch, batch_idx):
        # HACK: Move to cpu in order to hold more batches in memory
        # This is only needed when trainers need in-memory
        # EvaluationDataPages of the full evaluation dataset
        return EvaluationDataPage.create_from_training_batch(batch, self).cpu()

    def validation_epoch_end(self, valid_step_outputs):
        # As explained in the comments to the validation_step function in
        # pytorch_lightning/core/lightning.py, this function is generally used as follows:
        # val_outs = []
        # for val_batch in val_data:
        #     out = validation_step(val_batch)
        #     val_outs.append(out)
        # validation_epoch_end(val_outs)

        # The input arguments of validation_epoch_end() is a list of EvaluationDataPages,
        # which matches the way it is used in gather_eval_data() above.

        eval_data = self.gather_eval_data(valid_step_outputs)
        if eval_data and eval_data.mdp_id is not None:
            cpe_details = self.evaluator.evaluate_post_training(eval_data)
            self.reporter.log(cpe_details=cpe_details)
