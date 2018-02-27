#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from typing import Dict, Optional

import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python import workspace
from caffe2.python.model_helper import ModelHelper

import logging
logger = logging.getLogger(__name__)

from ml.rl.caffe_utils import C2
from ml.rl.preprocessing.normalization import (
    NormalizationParameters, get_num_output_features
)
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.discrete_action_predictor import DiscreteActionPredictor
from ml.rl.training.model_update_helper import AddParameterUpdateOps
from ml.rl.training.ml_trainer import GenerateLossOps
from ml.rl.training.ml_trainer import MLTrainer, MakeForwardPassOps
from ml.rl.training.rl_trainer import RLTrainer


class DiscreteActionTrainer(RLTrainer):
    # Set to a very large negative number.  Guaranteed to be worse than any
    #     legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9

    def __init__(
        self,
        parameters: DiscreteActionModelParameters,
        normalization_parameters: Dict[int, NormalizationParameters],
    ) -> None:
        self._actions = parameters.actions if parameters.actions is not None else []
        self.reward_shape = {}  # type: Dict[int, float]
        if parameters.rl.reward_boost is not None and self._actions is not None:
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                self.reward_shape[i] = parameters.rl.reward_boost[k]
        self.state_normalization_parameters = normalization_parameters
        num_features = get_num_output_features(normalization_parameters)
        parameters.training.layers[0] = num_features
        parameters.training.layers[-1] = self.num_actions

        RLTrainer.__init__(self, parameters)

        self._create_all_q_score_net()

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def get_possible_next_actions(self):
        return 'possible_next_actions'

    def _setup_initial_blobs(self):
        self.input_dim = self.num_features
        self.output_dim = self.num_actions

        MLTrainer._setup_initial_blobs(self)

    def _create_all_q_score_net(self) -> None:
        self.all_q_score_model = ModelHelper(
            name="all_q_score_" + self.model_id
        )
        C2.set_model(self.all_q_score_model)
        self.all_q_score_output = self.get_q_values_all_actions('states', True)
        workspace.RunNetOnce(self.all_q_score_model.param_init_net)
        workspace.CreateNet(self.all_q_score_model.net)
        C2.set_model(None)

    def update_model(
        self,
        states: str,
        actions: str,
        q_vals_target: str,
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
            row contains the one-hotted representation of the ith action.
        :param q_vals_targets: Numpy array with shape (batch_size, 1). The ith
            row is the label to train against for the data from the ith transition.
        """
        model = C2.model()
        q_vals_target = C2.StopGradient(q_vals_target)
        output_blob = C2.NextBlob("train_output")
        MakeForwardPassOps(
            model,
            self.model_id,
            states,
            output_blob,
            self.weights,
            self.biases,
            self.activations,
            self.layers,
            self.dropout_ratio,
            False,
        )
        q_val_select = C2.ReduceBackSum(C2.Mul(output_blob, actions))
        q_values = C2.ExpandDims(q_val_select, dims=[1])

        self.loss_blob = GenerateLossOps(
            model,
            q_values,
            q_vals_target,
        )
        model.AddGradientOperators([self.loss_blob])
        for param in model.params:
            if param in model.param_to_grad:
                param_grad = model.param_to_grad[param]
                param_grad = C2.NanCheck(param_grad)
        AddParameterUpdateOps(
            model,
            optimizer_input=self.optimizer,
            base_learning_rate=self.learning_rate,
            gamma=self.gamma,
            policy=self.lr_policy,
        )

    def get_q_values(
        self,
        states: str,
        actions: str,
        use_target_network: bool,
    ) -> str:
        # actions and possible_next_actions are the same matrix, only that
        # actions is one-hot.  Because of this, we can call get_max_q_values.
        return self.get_max_q_values(
            states,
            actions,
            use_target_network,
        )

    def get_max_q_values(
        self,
        states: str,
        possible_actions: str,
        use_target_network: bool,
    ) -> str:
        """
        Takes in an array of states and outputs an array of the same shape
        whose ith entry = max_{pna} Q(state_i, pna).

        :param states: Numpy array with shape (batch_size, state_dim). Each
            row contains a representation of a state.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        :use_target_network: Boolean that indicates whether or not to use this
            trainer's TargetNetwork to compute Q values.
        """
        q_values = self.get_q_values_all_actions(states, use_target_network)

        # Set the q values of impossible actions to a very large negative
        #    number.
        inverse_pna = C2.ConstantFill(
            possible_actions,
            value=1.0,
        )
        inverse_pna = C2.Sub(
            inverse_pna,
            possible_actions,
        )
        inverse_pna = C2.Mul(
            inverse_pna,
            self.ACTION_NOT_POSSIBLE_VAL,
            broadcast=1,
        )
        q_values = C2.Add(q_values, inverse_pna)

        q_values_max = C2.ReduceBackMax(
            q_values,
            num_reduce_dims=1,
        )
        return C2.ExpandDims(q_values_max, dims=[1])

    def get_q_values_all_actions(
        self,
        states: str,
        use_target_network: bool,
    ) -> str:
        """
        Takes in a set of states and runs the test Q Network on them.

        Creates Q(states, actions), a blob with shape (batch_size, action_dim).
        Q(states, actions)[i][j] is an approximation of Q*(states[i], action_j).
        Note that action_j takes on every possible action (of which there are
        self.action_dim_. Stores blob in self.output_blob and returns its value.

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        :use_target_network: Boolean that indicates whether or not to use this
            trainer's TargetNetwork to compute Q values.
        """
        if use_target_network:
            return self.target_network.target_values(states)
        else:
            all_q_values = C2.NextBlob("all_q_values")
            MakeForwardPassOps(
                C2.model(),
                self.model_id + "_score",
                states,
                all_q_values,
                self.weights,
                self.biases,
                self.activations,
                self.layers,
                self.dropout_ratio,
                True,
            )
            return all_q_values

    def _create_reward_train_net(self) -> None:
        self.reward_train_model = ModelHelper(
            name="reward_train_" + self.model_id
        )
        C2.set_model(self.reward_train_model)
        if self.reward_shape is not None:
            for action_index, boost in self.reward_shape.items():
                action_boost = C2.Mul(
                    C2.Slice(
                        'actions',
                        starts=[0, action_index],
                        ends=[-1, action_index + 1],
                    ),
                    boost,
                    broadcast=1,
                )
                C2.net().Sum(['rewards', action_boost], ['rewards'])
        self.update_model('states', 'actions', 'rewards')
        workspace.RunNetOnce(self.reward_train_model.param_init_net)
        workspace.CreateNet(self.reward_train_model.net)
        C2.set_model(None)

    def _create_rl_train_net(self) -> None:
        self.rl_train_model = ModelHelper(name="rl_train_" + self.model_id)
        C2.set_model(self.rl_train_model)

        if self.reward_shape is not None:
            for action_index, boost in self.reward_shape.items():
                action_boost = C2.Mul(
                    C2.Slice(
                        'actions',
                        starts=[0, action_index],
                        ends=[-1, action_index + 1],
                    ),
                    boost,
                    broadcast=1,
                )
                C2.net().Sum(['rewards', action_boost], ['rewards'])

        if self.maxq_learning:
            next_q_values = self.get_max_q_values(
                'next_states',
                self.get_possible_next_actions(),
                True,
            )
        else:
            next_q_values = self.get_q_values(
                'next_states', 'next_actions', True
            )

        q_vals_target = C2.Add(
            'rewards',
            C2.Mul(
                C2.Mul(
                    C2.Cast('not_terminals',
                            to=caffe2_pb2.TensorProto.FLOAT),  # type: ignore
                    self.rl_discount_rate,
                    broadcast=1,
                ),
                next_q_values
            )
        )

        self.update_model('states', 'actions', q_vals_target)
        workspace.RunNetOnce(self.rl_train_model.param_init_net)
        workspace.CreateNet(self.rl_train_model.net)
        C2.set_model(None)

    def predictor(self) -> DiscreteActionPredictor:
        """
        Builds a DiscreteActionPredictor using the MLTrainer underlying this
        DiscreteActionTrainer.
        """
        return DiscreteActionPredictor.export(
            self,
            self._actions,
            self.state_normalization_parameters,
        )

    def get_policy(
        self,
        states: np.ndarray,
        possible_next_actions: Optional[np.ndarray],
    ) -> int:
        """
        Returns the index of the action with the highest approximated q-value
        for the given state.

        :param state: A Numpy array of shape (N, state_dim) containing a
            set of normalized state vectors.
        """
        assert self.q_score_model is not None
        workspace.FeedBlob('states', states)
        if possible_next_actions is not None:
            workspace.FeedBlob('actions', possible_next_actions)
            workspace.RunNetOnce(self.q_score_model.net)
            q_values = workspace.FetchBlob(self.q_score_output)
        else:
            workspace.RunNetOnce(self.all_q_score_model.net)
            q_values = workspace.FetchBlob(self.all_q_score_output)
        return np.argmax(q_values, axis=1).reshape(-1, 1)
