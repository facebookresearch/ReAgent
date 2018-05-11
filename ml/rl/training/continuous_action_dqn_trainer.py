#!/usr/bin/env python3

from typing import Dict

import logging
logger = logging.getLogger(__name__)

import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python import workspace
from caffe2.python.model_helper import ModelHelper

from ml.rl.caffe_utils import C2, StackedArray
from ml.rl.preprocessing.normalization import (
    NormalizationParameters, get_num_output_features
)
from ml.rl.thrift.core.ttypes import (
    ContinuousActionModelParameters, AdditionalFeatureTypes
)
from ml.rl.training.continuous_action_dqn_predictor import (
    ContinuousActionDQNPredictor
)
from ml.rl.training.rl_trainer import RLTrainer, DEFAULT_ADDITIONAL_FEATURE_TYPES


class ContinuousActionDQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters: ContinuousActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        additional_feature_types:
        AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES
    ) -> None:
        self._additional_feature_types = additional_feature_types
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters
        num_features = get_num_output_features(
            state_normalization_parameters
        ) + get_num_output_features(action_normalization_parameters)

        # ensure state and action IDs have no intersection
        overlapping_features = (
            set(state_normalization_parameters.keys()) &
            set(action_normalization_parameters.keys())
        )
        assert (
            len(overlapping_features) == 0
        ), "There are some overlapping state and action features: " + str(
            overlapping_features
        )

        parameters.training.layers[0] = num_features
        parameters.training.layers[-1] = 1

        RLTrainer.__init__(self, parameters)

        self._create_internal_policy_net()

    def _create_internal_policy_net(self) -> None:
        self.internal_policy_model = ModelHelper(
            name="q_score_" + self.model_id
        )
        C2.set_model(self.internal_policy_model)
        self.internal_policy_output = C2.FlattenToVec(
            self.get_q_values('states', 'actions', False)
        )
        workspace.RunNetOnce(self.internal_policy_model.param_init_net)
        workspace.CreateNet(self.internal_policy_model.net)
        C2.set_model(None)

    def get_possible_next_actions(self):
        return StackedArray(
            'possible_next_actions_lengths',
            'possible_next_actions',
        )

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
            row is a representation of the ith transition's action.
        :param q_vals_targets: Numpy array with shape (batch_size, 1). The ith
            row is the label to train against for the data from the ith transition.
        """
        model = C2.model()
        q_vals_target = C2.StopGradient(q_vals_target)
        q_values = C2.NextBlob("train_output")
        state_action_pairs, _ = C2.Concat(states, actions, axis=1)
        self.ml_trainer.make_forward_pass_ops(
            model,
            state_action_pairs,
            q_values,
            False,
        )

        self.loss_blob = self.ml_trainer.generateLossOps(
            model,
            q_values,
            q_vals_target,
        )
        model.AddGradientOperators([self.loss_blob])
        for param in model.params:
            if param in model.param_to_grad:
                param_grad = model.param_to_grad[param]
                param_grad = C2.NanCheck(param_grad)
        self.ml_trainer.addParameterUpdateOps(model)

    def get_q_values(
        self,
        states: str,
        actions: str,
        use_target_network: bool,
    ) -> str:
        state_action_pairs, _ = C2.Concat(states, actions, axis=1)
        q_values = C2.NextBlob("q_values")
        if use_target_network:
            self.target_network.make_forward_pass_ops(
                C2.model(),
                state_action_pairs,
                q_values,
                True,
            )
        else:
            self.ml_trainer.make_forward_pass_ops(
                C2.model(),
                state_action_pairs,
                q_values,
                True,
            )
        return q_values

    def get_max_q_values(
        self,
        next_states: str,
        possible_next_actions: StackedArray,
        use_target_network: bool,
    ) -> str:
        """
        Takes in an array of next_states and outputs an array of the same shape
        whose ith entry = max_{pna} Q(state_i, pna). Uses target network for
        Q(state_i, pna) approximation.

        :param next_states: Blob containing state features.  Each
            row contains a representation of a state.
        :param possible_next_actions: List of sets of possible next actions. The
            ith element of this list is a matrix PNA_i such that PNA_i[j] is the
            parametric representation of the jth possible action from the ith
            next_state. These have not been normalized.
        """

        stacked_states = C2.LengthsTile(
            next_states, possible_next_actions.lengths
        )
        all_q_values = self.get_q_values(
            stacked_states,
            possible_next_actions.values,
            use_target_network,
        )
        max_q_values = C2.LengthsMax(
            all_q_values,
            possible_next_actions.lengths,
        )
        return max_q_values

    def _create_reward_train_net(self) -> None:
        self.reward_train_model = ModelHelper(
            name="reward_train_" + self.model_id
        )
        C2.set_model(self.reward_train_model)
        self.update_model('states', 'actions', 'rewards')
        workspace.RunNetOnce(self.reward_train_model.param_init_net)
        workspace.CreateNet(self.reward_train_model.net)
        C2.set_model(None)

    def _create_rl_train_net(self) -> None:
        self.rl_train_model = ModelHelper(name="rl_train_" + self.model_id)
        C2.set_model(self.rl_train_model)

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

        discount_blob = C2.ConstantFill("time_diff", value=self.rl_discount_rate)
        time_diff_adjusted_discount_blob = C2.Pow(
            discount_blob,
            C2.Cast("time_diff", to=caffe2_pb2.TensorProto.FLOAT)
        )

        q_vals_target = C2.Add(
            "rewards",
            C2.Mul(
                C2.Mul(
                    C2.Cast(
                        "not_terminals", to=caffe2_pb2.TensorProto.FLOAT
                    ),  # type: ignore
                    time_diff_adjusted_discount_blob,
                    broadcast=1,
                ),
                next_q_values,
            ),
        )

        self.update_model('states', 'actions', q_vals_target)
        workspace.RunNetOnce(self.rl_train_model.param_init_net)
        workspace.CreateNet(self.rl_train_model.net)
        C2.set_model(None)

    def predictor(self) -> ContinuousActionDQNPredictor:
        """
        Builds a ContinuousActionPredictor using the MLTrainer underlying this
        ContinuousActionTrainer.
        """
        return ContinuousActionDQNPredictor.export(
            self,
            self.state_normalization_parameters,
            self.action_normalization_parameters,
            self._additional_feature_types.int_features,
        )
