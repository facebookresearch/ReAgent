#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Optional, List
import numpy as np

from caffe2.python import workspace

import logging
logger = logging.getLogger(__name__)

from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.ml_conv_trainer import MLConvTrainer
from ml.rl.training.training_data_page import TrainingDataPage
from ml.rl.training.ml_trainer import GenerateLossOps


class DiscreteActionConvTrainer(MLConvTrainer):
    ACTION_NOT_POSSIBLE_VAL = -1e20

    def __init__(
        self,
        parameters: DiscreteActionModelParameters,
        init_height: int,
        init_width: int,
        num_input_channels: int,
        action_dim: int,
        conv_dims: List[int],
        conv_height_kernels: List[int],
        conv_width_kernels: List[int],
        pool_kernels_strides: List[int],
        pool_types: List[str]
    ) -> None:
        self.action_dim = action_dim
        self.num_actions = action_dim
        assert parameters.training.layers[0] in [None, -1, 1],\
            "Let MLConvTrainer set layers[0]"

        if parameters.training.layers[-1] in [None, -1, 1]:
            parameters.training.layers[-1] = action_dim

        assert parameters.training.layers[-1] == self.num_actions,\
            "Set layers[-1] to a the number of actions or a default placeholder value"

        MLConvTrainer.__init__(
            self, "ml_conv_trainer", parameters.training, init_height, init_width,
            conv_dims, conv_height_kernels, conv_width_kernels, pool_kernels_strides,
            pool_types, num_input_channels
        )

        self.reward_burnin = parameters.rl.reward_burnin
        self.maxq_learning = parameters.rl.maxq_learning
        self.rl_discount_rate = parameters.rl.gamma

        self.training_iteration = 0

    def stream_tdp(
        self, tdp: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:
        self.train(
            tdp.states, tdp.actions, tdp.rewards, tdp.next_states, tdp.next_actions,
            tdp.not_terminals, tdp.possible_next_actions
        )

    def _reshape_states(self, inputs):
        """
        Reshapes NHWC into NCHW format.
        """
        n, h, w, c = inputs.shape
        return inputs.reshape((n, c, h, w))

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_actions: Optional[np.ndarray],
        not_terminals: np.ndarray,
        possible_next_actions: np.ndarray,
    ) -> None:
        batch_size = states.shape[0]
        assert actions.shape == (batch_size, self.num_actions)

        if not_terminals.ndim == 1:
            not_terminals = not_terminals.reshape(-1, 1)

        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)

        assert not_terminals.shape == (batch_size, 1)
        assert rewards.shape == (batch_size, 1)

        states = self._reshape_states(states)
        next_states = self._reshape_states(next_states)

        q_vals_target = np.copy(rewards)
        if self.training_iteration >= self.reward_burnin:
            if self.training_iteration == self.reward_burnin:
                logger.info(
                    "Minibatch number == reward_burnin. Starting RL updates."
                )
            next_q_values = self.get_max_q_values(
                next_states, possible_next_actions
            )

            q_vals_target += not_terminals * self.rl_discount_rate * next_q_values

        self.update_model(states, actions, q_vals_target)

        self.training_iteration += 1

    def _setup_initial_blobs(self):
        self.action_blob = "action"
        workspace.FeedBlob(self.action_blob, np.zeros(1, dtype=np.float32))

        MLConvTrainer._setup_initial_blobs(self)

    def _generate_train_model_loss(self):
        q_val_select = self.train_model.net.Mul(
            [self.output_blob, self.action_blob],
        ).ReduceBackSum()
        q_values = self.train_model.net.ExpandDims(q_val_select, dims=[1])

        GenerateLossOps(
            self.train_model, self.model_id + "_train", self.labels_blob, q_values,
            self.loss_blob
        )

    def update_model(
        self, states: np.ndarray, actions: np.ndarray, q_vals_target: np.ndarray
    ) -> None:
        workspace.FeedBlob(self.action_blob, actions)
        self.train_batch(states, q_vals_target)

    def get_max_q_values(
        self,
        next_states: np.ndarray,
        possible_next_actions: Optional[np.ndarray] = None,
        use_target_network: Optional[bool] = True
    ) -> np.ndarray:
        q_values = self.get_q_values_all_actions(
            next_states, use_target_network
        )

        if possible_next_actions is not None:
            mask = np.multiply(
                np.logical_not(possible_next_actions),
                self.ACTION_NOT_POSSIBLE_VAL
            )
            q_values += mask

        return np.max(q_values, axis=1, keepdims=True)

    def get_q_values_all_actions(
        self, states: np.ndarray, use_target_network: Optional[bool] = True
    ) -> np.ndarray:
        return self.score(states)

    def get_policy(self, state: np.ndarray) -> int:
        inputs = self._reshape_states(np.array([state], dtype=np.float32))
        q_values = self.get_q_values_all_actions(inputs, False)
        return np.argmax(q_values[0])
