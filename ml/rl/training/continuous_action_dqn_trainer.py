#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from typing import List, Dict, Optional

from caffe2.python import core, workspace
import caffe2.proto.caffe2_pb2 as caffe2_pb2

import logging
logger = logging.getLogger(__name__)

from ml.rl.preprocessing.normalization import NormalizationParameters,\
    get_num_output_features
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet, \
    sort_features_by_normalization
from ml.rl.thrift.core.ttypes import ContinuousActionModelParameters
from ml.rl.training.continuous_action_dqn_predictor import\
    ContinuousActionDQNPredictor
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.rl_trainer import RLTrainer
from ml.rl.training.training_data_page import TrainingDataPage


class ContinuousActionDQNTrainer(RLTrainer):
    def __init__(
        self,
        state_normalization_parameters: Dict[str, NormalizationParameters],
        action_normalization_parameters: Dict[str, NormalizationParameters],
        parameters: ContinuousActionModelParameters,
        skip_normalization: Optional[bool] = False
    ) -> None:
        self._action_features, _ = sort_features_by_normalization(
            action_normalization_parameters
        )
        self.num_unprocessed_action_features = len(self._action_features)
        self.num_processed_action_features = get_num_output_features(
            action_normalization_parameters
        )

        self.num_processed_state_features = get_num_output_features(
            state_normalization_parameters
        )

        if parameters.training.layers[0] is None or\
           parameters.training.layers[0] == -1:
            parameters.training.layers[0] = self.num_state_features +\
                self.num_action_features

        assert parameters.training.layers[-1] == 1, "Set layers[-1] to 1"

        self._action_normalization_parameters = action_normalization_parameters
        RLTrainer.__init__(
            self, state_normalization_parameters, parameters, skip_normalization
        )
        print(action_normalization_parameters)

        self._prepare_action_normalization()

    def get_action_features(self) -> List[str]:
        return self._action_features

    @property
    def num_state_features(self) -> int:
        return self.num_processed_state_features

    @property
    def num_action_features(self) -> int:
        return self.num_processed_action_features

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.skip_normalization:
            return actions
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            workspace.FeedBlob(self.action_input_matrix, actions)
            workspace.RunNetOnce(self.action_norm_net)
            return workspace.FetchBlob(self.action_preprocessed_matrix)

    def _prepare_action_normalization(self):
        """
        Sets up operators for action normalization net.
        """
        if self.skip_normalization:
            return
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            self.action_norm_net = core.Net("action_norm_net")
            self.action_preprocessor = PreprocessorNet(
                self.action_norm_net, True
            )
            self.action_input_matrix = 'action_input_matrix'
            self.action_preprocessed_matrix, _ = \
                self.action_preprocessor.normalize_dense_matrix(
                    self.action_input_matrix, self._action_features,
                    self._action_normalization_parameters, 'action'
                )

    def _setup_initial_blobs(self):
        self.input_dim = self.num_state_features + self.num_action_features
        self.output_dim = 1

        MLTrainer._setup_initial_blobs(self)

    def _convert_to_net_inputs(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """
        Shapes states and actions into an input format compatible with this
        Trainer's underlying net and its target network's net.
        """
        return np.concatenate([states, actions], axis=1)

    def stream_tdp(
        self, tdp: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:
        """
        Loads a large batch of transitions from a page of training data. This
        batch will further be broken down into minibatches for training.

        :param tdp: TrainingDataPage object that supplies transitions.
        :param evaluator: Evaluator object to record TD and compute MC losses.
        """
        not_terminals = tdp.not_terminals
        if tdp.not_terminals is None:
            not_terminals = np.array(
                [pna.shape[0] != 0 for pna in tdp.possible_next_actions],
                dtype=np.bool
            )

        self.stream(
            tdp.states, tdp.actions, tdp.rewards, tdp.next_states,
            tdp.next_actions, not_terminals, tdp.possible_next_actions,
            tdp.reward_timelines, evaluator
        )

    def _validate_train_inputs(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
        next_states: np.ndarray, next_actions: Optional[np.ndarray],
        not_terminals: np.ndarray,
        possible_next_actions: Optional[List[np.ndarray]]
    ) -> None:
        batch_size = states.shape[0]
        assert actions.shape == (batch_size, self.num_action_features)
        if next_actions is not None:
            assert next_actions.shape == (batch_size, self.num_action_features)
        if possible_next_actions is not None:
            assert len(possible_next_actions) == batch_size
            for pna in possible_next_actions:
                if pna.shape[0] > 0:
                    assert pna.shape[1] == self.num_unprocessed_action_features

    def update_model(
        self, states: np.ndarray, actions: np.ndarray, q_vals_target: np.ndarray
    ) -> None:
        """
        Takes in states, actions, and target q values. Updates the model:

            Runs the forward pass, computing Q(states, actions).
                Q(states, actions)[i][j] is an approximation of Q*(states[i], action_j).
            Comptutes Loss of Q(states, actions) with respect to q_vals_targets
            Updates Q Network's weights according to loss and optimizer

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row is a representation of the ith transition's action.
        :param q_vals_targets: Numpy array with shape (batch_size, 1). The ith
            row is the label to train against for the data from the ith transition.
        """
        inputs = self._convert_to_net_inputs(states, actions)
        self.train_batch(inputs, q_vals_target)

    def get_max_q_values(
        self, next_states: np.ndarray, possible_next_actions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Takes in an array of next_states and outputs an array of the same shape
        whose ith entry = max_{pna} Q(state_i, pna). Uses target network for
        Q(state_i, pna) approximation.

        :param next_states: Numpy array with shape (batch_size, state_dim). Each
            row contains a representation of a state.
        :param possible_next_actions: List of sets of possible next actions. The
            ith element of this list is a matrix PNA_i such that PNA_i[j] is the
            parametric representation of the jth possible action from the ith
            next_state. These have not been normalized.
        """
        total_size = 0
        sizes = []
        for i in range(len(next_states)):
            num_possible_actions = possible_next_actions[i].shape[0]
            sizes.append(num_possible_actions)
            total_size += num_possible_actions

        normalized_stacked_pna = self._normalize_actions(
            np.row_stack(
                list(
                    filter(lambda pna: pna.shape[0] > 0, possible_next_actions)
                )
            )
        )

        num_total_features = self.num_state_features + self.num_action_features
        inputs_to_score = np.zeros(
            [total_size, num_total_features], dtype=np.float32
        )
        cursor = 0
        for i in range(len(next_states)):
            num_possible_actions = sizes[i]
            if num_possible_actions == 0:
                continue
            cursor_end = cursor + num_possible_actions
            possible_actions = normalized_stacked_pna[cursor:cursor_end]
            inputs_to_score[cursor:cursor_end, 0:self.num_state_features] \
                = np.repeat(
                    next_states[i].reshape(1, self.num_state_features),
                    num_possible_actions,
                    axis=0)
            inputs_to_score[cursor:cursor_end,
                            self.num_state_features:num_total_features] = \
                            possible_actions
            cursor += num_possible_actions
        all_q_values = self.target_network.target_values(inputs_to_score)
        cursor = 0
        q_values = np.zeros([len(next_states), 1], dtype=np.float32)
        for i in range(len(next_states)):
            num_possible_actions = sizes[i]
            if num_possible_actions == 0:
                continue
            q_values[i, 0] = np.max(
                all_q_values[cursor:(cursor + num_possible_actions)]
            )
            cursor += num_possible_actions
        return q_values

    def get_q_values(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """
        Takes in a set of states and actions and returns Q(states, actions).

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row is a representation of the ith transition's action.
        """
        return self.score(self._convert_to_net_inputs(states, actions))

    def get_sarsa_values(
        self, next_states: np.ndarray, next_actions: np.ndarray
    ) -> np.ndarray:
        """
        Takes in a set of next_states and corresponding next_actions. For each
        (next_state_i, next_action_i) pair, calculates Q(next_state, next_action).
        Returns these q values in a Numpy array of shape (batch_size, 1).

        :param next_states: Numpy array with shape (batch_size, state_dim). The
            ith row is a representation of the ith transition's next_state.
        :param next_actions: Numpy array with shape (batch_size, action_dim).
            The ith row is a representation of the ith transition's next_action.
            Note that these are not normalized.
        """
        inputs = self._convert_to_net_inputs(next_states, next_actions)
        return self.target_network.target_values(inputs)

    def predictor(self) -> ContinuousActionDQNPredictor:
        """
        Builds a ContinuousActionPredictor using the MLTrainer underlying this
        ContinuousActionTrainer.
        """
        return ContinuousActionDQNPredictor.export(
            self, self._state_normalization_parameters,
            self._action_normalization_parameters
        )
