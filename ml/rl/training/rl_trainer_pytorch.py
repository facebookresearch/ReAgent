#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ml.rl.training.loss_reporter import LossReporter


logger = logging.getLogger(__name__)


class RLTrainer:
    # Q-value for action that is not possible. Guaranteed to be worse than any
    # legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9
    # Hack to mark legitimate 0 value q-values before pytorch sparse -> dense
    FINGERPRINT = 12345

    def __init__(
        self,
        parameters,
        use_gpu,
        metrics_to_score=None,
        actions: Optional[List[str]] = None,
    ):
        self.minibatch = 0
        self.parameters = parameters
        self.rl_temperature = float(parameters.rl.temperature)
        self.maxq_learning = parameters.rl.maxq_learning
        self.gamma = parameters.rl.gamma
        self.tau = parameters.rl.target_update_rate
        self.use_seq_num_diff_as_time_diff = parameters.rl.use_seq_num_diff_as_time_diff
        self.time_diff_unit_length = parameters.rl.time_diff_unit_length
        self.tensorboard_logging_freq = parameters.rl.tensorboard_logging_freq
        self.multi_steps = parameters.rl.multi_steps
        self.calc_cpe_in_training = parameters.evaluation.calc_cpe_in_training

        if parameters.rl.q_network_loss == "mse":
            self.q_network_loss = F.mse_loss
        elif parameters.rl.q_network_loss == "huber":
            self.q_network_loss = F.smooth_l1_loss
        else:
            raise Exception(
                "Q-Network loss type {} not valid loss.".format(
                    parameters.rl.q_network_loss
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
            self.dtype = torch.cuda.FloatTensor
            self.dtypelong = torch.cuda.LongTensor
        else:
            logger.info("NOT Using GPU: GPU not requested or not available.")
            self.use_gpu = False
            self.dtype = torch.FloatTensor
            self.dtypelong = torch.LongTensor

        self.loss_reporter = LossReporter(actions)

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

    def _maybe_soft_update(
        self, network, target_network, tau, minibatches_per_step
    ) -> None:
        if self.minibatch % minibatches_per_step != 0:
            return
        return self._soft_update(network, target_network, tau)

    def _maybe_run_optimizer(self, optimizer, minibatches_per_step) -> None:
        if self.minibatch % minibatches_per_step != 0:
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad /= minibatches_per_step
        optimizer.step()
        optimizer.zero_grad()

    def train(self, training_samples, evaluator=None) -> None:
        raise NotImplementedError()

    def state_dict(self):
        return {c: getattr(self, c).state_dict() for c in self.warm_start_components()}

    def load_state_dict(self, state_dict):
        for c in self.warm_start_components():
            getattr(self, c).load_state_dict(state_dict[c])

    def warm_start_components(self):
        """
        The trainer should specify what members to save and load
        """
        raise NotImplementedError

    def internal_prediction(self, input):
        """ Q-network forward pass method for internal domains.
        :param input input to network
        """
        self.q_network.eval()
        with torch.no_grad():
            input = torch.from_numpy(np.array(input)).type(self.dtype)
            q_values = self.q_network(input)
        self.q_network.train()
        return q_values.cpu().data.numpy()

    def internal_reward_estimation(self, input):
        """ Reward-network forward pass for internal domains. """
        self.reward_network.eval()
        with torch.no_grad():
            input = torch.from_numpy(np.array(input)).type(self.dtype)
            reward_estimates = self.reward_network(input)
        self.reward_network.train()
        return reward_estimates.cpu().data.numpy()


def rescale_torch_tensor(
    tensor: torch.tensor,
    new_min: torch.tensor,
    new_max: torch.tensor,
    prev_min: torch.tensor,
    prev_max: torch.tensor,
):
    """
    Rescale column values in N X M torch tensor to be in new range.
    Each column m in input tensor will be rescaled from range
    [prev_min[m], prev_max[m]] to [new_min[m], new_max[m]]
    """
    assert tensor.shape[1] == new_min.shape[1] == new_max.shape[1]
    assert tensor.shape[1] == prev_min.shape[1] == prev_max.shape[1]
    prev_range = prev_max - prev_min
    new_range = new_max - new_min
    return ((tensor - prev_min) / prev_range) * new_range + new_min
