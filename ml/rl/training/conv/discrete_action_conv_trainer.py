#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Optional
import numpy as np

from caffe2.python import workspace

from ml.rl.thrift.core.ttypes import DiscreteActionConvModelParameters
from ml.rl.training.discrete_action_predictor import DiscreteActionPredictor
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.conv.ml_conv_trainer import MLConvTrainer
from ml.rl.training.conv.rl_conv_trainer import RLConvTrainer
from ml.rl.training.training_data_page import TrainingDataPage


class DiscreteActionConvTrainer(RLConvTrainer, DiscreteActionTrainer):
    def __init__(
        self,
        parameters: DiscreteActionConvModelParameters,
    ) -> None:
        fc_parameters = parameters.fc_parameters
        cnn_parameters = parameters.cnn_parameters

        # To convince the typechecker that conv_dims exists
        if (
            cnn_parameters is None
            or cnn_parameters.conv_dims is None
            or fc_parameters is None
            or parameters.img_height is None
            or parameters.img_width is None
            or parameters.num_input_channels is None
        ):
            raise Exception("Please supply all arguments")

        assert fc_parameters.training.layers[0] in [None, -1],\
            "Let MLConvTrainer set layers[0]"

        assert cnn_parameters.conv_dims[0] in [None, -1],\
            "Let DiscreteActionConvTrainer set conv_dims[0]"

        cnn_parameters.conv_dims[0] = parameters.num_input_channels

        self._actions = fc_parameters.actions
        if fc_parameters.training.layers[-1] in [None, -1]:
            fc_parameters.training.layers[-1] = self.num_actions

        assert fc_parameters.training.layers[-1] == self.num_actions,\
            "Set layers[-1] to a the number of actions or a default placeholder value"

        RLConvTrainer.__init__(
            self, fc_parameters, cnn_parameters, parameters.img_height,
            parameters.img_width
        )

    @property
    def num_state_features(self) -> int:
        raise NotImplementedError()

    def stream_tdp(
        self, tdp: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:
        self.stream(
            self._reshape_states(tdp.states), tdp.actions, tdp.rewards,
            self._reshape_states(tdp.next_states), tdp.next_actions,
            tdp.not_terminals, tdp.possible_next_actions, tdp.reward_timelines,
            evaluator
        )

    def _reshape_states(self, inputs):
        """
        Reshapes NHWC into NCHW format.
        """
        n, h, w, c = inputs.shape
        return inputs.reshape((n, c, h, w))

    def _validate_train_inputs(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_actions: Optional[np.ndarray],
        not_terminals: np.ndarray,
        possible_next_actions: np.ndarray,
    ):
        batch_size = states.shape[0]
        assert actions.shape == (batch_size, self.num_actions)

    def _setup_initial_blobs(self):
        self.action_blob = "action"
        workspace.FeedBlob(self.action_blob, np.zeros(1, dtype=np.float32))

        MLConvTrainer._setup_initial_blobs(self)

    def get_policy(self, state: np.ndarray) -> int:
        inputs = self._reshape_states(np.array([state], dtype=np.float32))
        q_values = self.get_q_values_all_actions(inputs, False)
        return np.argmax(q_values[0])

    def predictor(self) -> DiscreteActionPredictor:
        raise NotImplementedError()
