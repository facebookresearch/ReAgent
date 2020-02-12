#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.parameters import EvaluationParameters, OptimizerParameters, RLParameters
from ml.rl.torch_utils import masked_softmax
from ml.rl.training.loss_reporter import LossReporter
from ml.rl.training.trainer import Trainer


logger = logging.getLogger(__name__)


class RLTrainer(Trainer):
    # Q-value for action that is not possible. Guaranteed to be worse than any
    # legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9
    # Hack to mark legitimate 0 value q-values before pytorch sparse -> dense
    FINGERPRINT = 12345

    def __init__(
        self,
        rl_parameters: RLParameters,
        use_gpu: bool,
        metrics_to_score=None,
        actions: Optional[List[str]] = None,
        evaluation_parameters: Optional[EvaluationParameters] = None,
    ) -> None:
        self.minibatch = 0
        self.minibatch_size: Optional[int] = None
        self.minibatches_per_step: Optional[int] = None
        self.rl_parameters = rl_parameters
        self.rl_temperature = float(rl_parameters.temperature)
        self.maxq_learning = rl_parameters.maxq_learning
        self.gamma = rl_parameters.gamma
        self.tau = rl_parameters.target_update_rate
        self.use_seq_num_diff_as_time_diff = rl_parameters.use_seq_num_diff_as_time_diff
        self.time_diff_unit_length = rl_parameters.time_diff_unit_length
        self.tensorboard_logging_freq = rl_parameters.tensorboard_logging_freq
        self.multi_steps = rl_parameters.multi_steps
        self.calc_cpe_in_training = (
            evaluation_parameters and evaluation_parameters.calc_cpe_in_training
        )

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

        cuda_available = torch.cuda.is_available()
        logger.info("CUDA availability: {}".format(cuda_available))
        if use_gpu and cuda_available:
            logger.info("Using GPU: GPU requested and available.")
            self.use_gpu = True
            self.device = torch.device("cuda")  # type: ignore
        else:
            logger.info("NOT Using GPU: GPU not requested or not available.")
            self.use_gpu = False
            self.device = torch.device("cpu")  # type: ignore

        self.loss_reporter = LossReporter(actions)
        self._actions = actions

    @property
    def num_actions(self) -> int:
        assert self._actions is not None, "Not a discrete action DQN"
        return len(self._actions)

    def _initialize_cpe(
        self,
        parameters,
        reward_network,
        q_network_cpe,
        q_network_cpe_target,
        cpe_optimizer_parameters: OptimizerParameters,
    ) -> None:
        if self.calc_cpe_in_training:
            optimizer_func = self._get_optimizer_func(
                cpe_optimizer_parameters.optimizer
            )
            assert reward_network is not None, "reward_network is required for CPE"
            self.reward_network = reward_network
            self.reward_network_optimizer = optimizer_func(
                self.reward_network.parameters(),
                lr=cpe_optimizer_parameters.learning_rate,
                weight_decay=cpe_optimizer_parameters.l2_decay,
            )
            assert (
                q_network_cpe is not None and q_network_cpe_target is not None
            ), "q_network_cpe and q_network_cpe_target are required for CPE"
            self.q_network_cpe = q_network_cpe
            self.q_network_cpe_target = q_network_cpe_target
            self.q_network_cpe_optimizer = optimizer_func(
                self.q_network_cpe.parameters(),
                lr=cpe_optimizer_parameters.learning_rate,
            )
            num_output_nodes = len(self.metrics_to_score) * self.num_actions
            self.reward_idx_offsets = torch.arange(
                0,
                num_output_nodes,
                self.num_actions,
                device=self.device,
                dtype=torch.long,
            )
        else:
            self.reward_network = None

    def _set_optimizer(self, optimizer_name):
        self.optimizer_func = self._get_optimizer_func(optimizer_name)

    def _get_optimizer(self, network, param):
        return self._get_optimizer_func(param.optimizer)(
            network.parameters(), lr=param.learning_rate, weight_decay=param.l2_decay
        )

    def _get_optimizer_func(self, optimizer_name):
        if optimizer_name == "ADAM":
            return torch.optim.Adam
        elif optimizer_name == "SGD":
            return torch.optim.SGD
        else:
            raise NotImplementedError(
                "{} optimizer not implemented".format(optimizer_name)
            )

    @torch.no_grad()  # type: ignore
    def _soft_update(self, network, target_network, tau) -> None:
        """ Target network update logic as defined in DDPG paper
        updated_params = tau * network_params + (1 - tau) * target_network_params
        :param network network with parameters to include in soft update
        :param target_network target network with params to soft update
        :param tau hyperparameter to control target tracking speed
        """
        for t_param, param in zip(target_network.parameters(), network.parameters()):
            if t_param is param:
                # Skip soft-updating when the target network shares the parameter with
                # the network being train.
                continue
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)

    @torch.no_grad()  # type: ignore
    def _maybe_soft_update(
        self, network, target_network, tau, minibatches_per_step
    ) -> None:
        if self.minibatch % minibatches_per_step != 0:
            return
        self._soft_update(network, target_network, tau)

    def _maybe_run_optimizer(self, optimizer, minibatches_per_step) -> None:
        if self.minibatch % minibatches_per_step != 0:
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad /= minibatches_per_step
        optimizer.step()
        optimizer.zero_grad()

    @torch.no_grad()  # type: ignore
    def internal_prediction(self, input):
        """ Q-network forward pass method for internal domains.
        :param input input to network
        """
        self.q_network.eval()
        input = torch.from_numpy(np.array(input), device=self.device)
        q_values = self.q_network(input)
        self.q_network.train()
        return q_values.cpu()

    @torch.no_grad()  # type: ignore
    def internal_reward_estimation(self, input):
        """ Reward-network forward pass for internal domains. """
        self.reward_network.eval()
        input = torch.from_numpy(np.array(input), device=self.device)
        reward_estimates = self.reward_network(input)
        self.reward_network.train()
        return reward_estimates.cpu()

    @torch.no_grad()  # type: ignore
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
            return None, None, None

        if training_batch.extras.metrics is None:
            metrics_reward_concat_real_vals = training_batch.training_input.reward
        else:
            metrics_reward_concat_real_vals = torch.cat(
                (training_batch.training_input.reward, training_batch.extras.metrics),
                dim=1,
            )

        model_propensities_next_states = masked_softmax(
            all_next_action_scores,
            training_batch.training_input.possible_next_actions_mask
            if self.maxq_learning
            else training_batch.training_input.next_action,
            self.rl_temperature,
        )

        with torch.enable_grad():
            ######### Train separate reward network for CPE evaluation #############
            # FIXME: the reward network should be outputting a tensor,
            # not a q-value object
            reward_estimates = self.reward_network(states).q_values
            reward_estimates_for_logged_actions = reward_estimates.gather(
                1, self.reward_idx_offsets + logged_action_idxs
            )
            reward_loss = F.mse_loss(
                reward_estimates_for_logged_actions, metrics_reward_concat_real_vals
            )
            reward_loss.backward()
            self._maybe_run_optimizer(
                self.reward_network_optimizer, self.minibatches_per_step
            )

            ######### Train separate q-network for CPE evaluation #############
            metric_q_values = self.q_network_cpe(states).q_values.gather(
                1, self.reward_idx_offsets + logged_action_idxs
            )
            all_metrics_target_q_values = torch.chunk(
                self.q_network_cpe_target(next_states).q_values.detach(),
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
            metric_q_value_loss.backward()
            self._maybe_run_optimizer(
                self.q_network_cpe_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network_cpe,
            self.q_network_cpe_target,
            self.tau,
            self.minibatches_per_step,
        )

        model_propensities = masked_softmax(
            all_action_scores,
            training_batch.training_input.possible_actions_mask
            if self.maxq_learning
            else training_batch.training_input.action,
            self.rl_temperature,
        )
        model_rewards = reward_estimates[
            :,
            torch.arange(
                self.reward_idx_offsets[0],
                self.reward_idx_offsets[0] + self.num_actions,
            ),
        ]
        return reward_loss, model_rewards, model_propensities
