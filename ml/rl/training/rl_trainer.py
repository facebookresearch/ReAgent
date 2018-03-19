#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Dict, List, Optional, Union

import logging
logger = logging.getLogger(__name__)

import numpy as np

from caffe2.python import workspace
from caffe2.python.model_helper import ModelHelper

from ml.rl.caffe_utils import C2, StackedArray
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters,\
    ContinuousActionModelParameters
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.target_network import TargetNetwork
from ml.rl.training.training_data_page import TrainingDataPage
from ml.rl.training.evaluator import Evaluator

RL_TRAINER_MODEL_ID = "rl_trainer"


class RLTrainer(MLTrainer):
    def __init__(
        self,
        parameters: Union[DiscreteActionModelParameters,
                          ContinuousActionModelParameters],
    ) -> None:
        logger.info(str(parameters))

        assert parameters.training.layers[0] >= 0,\
            "Set layers[0] to a the number of features"

        self.num_features = parameters.training.layers[0]

        MLTrainer.__init__(self, RL_TRAINER_MODEL_ID, parameters.training)

        self.target_network = TargetNetwork(
            self, parameters.rl.target_update_rate
        )

        self.reward_burnin = parameters.rl.reward_burnin
        self.maxq_learning = parameters.rl.maxq_learning
        self.rl_discount_rate = parameters.rl.gamma
        self.rl_temperature = parameters.rl.temperature
        self.training_iteration = 0
        self.minibatch_size = parameters.training.minibatch_size
        self.parameters = parameters
        self.loss_blob: Optional[str] = None

        workspace.FeedBlob('states', np.array([0], dtype=np.float32))
        workspace.FeedBlob('actions', np.array([0], dtype=np.float32))
        workspace.FeedBlob('rewards', np.array([0], dtype=np.float32))
        workspace.FeedBlob('next_states', np.array([0], dtype=np.float32))
        workspace.FeedBlob('not_terminals', np.array([0], dtype=np.float32))
        workspace.FeedBlob('next_actions', np.array([0], dtype=np.float32))
        workspace.FeedBlob(
            'possible_next_actions', np.array([0], dtype=np.float32)
        )
        workspace.FeedBlob(
            'possible_next_actions_lengths', np.array([0], dtype=np.float32)
        )

        self.rl_train_model: Optional[ModelHelper] = None
        self.reward_train_model: Optional[ModelHelper] = None
        self.q_score_model: Optional[ModelHelper] = None
        self._create_reward_train_net()
        self._create_rl_train_net()
        self._create_q_score_net()
        assert self.rl_train_model is not None
        assert self.reward_train_model is not None
        assert self.q_score_model is not None

    def get_possible_next_actions(self):
        raise NotImplementedError()

    def get_max_q_values(
        self,
        next_states: str,
        possible_next_actions,
        use_target_network: bool,
    ) -> str:
        """
        Takes in an array of next_states and outputs an array of the same shape
        whose ith entry = max_{pna} Q(state_i, pna). Uses target network for
        Q(state_i, pna) approximation.

        :param next_states: Numpy array with shape (batch_size, state_dim). Each
            row contains a representation of a state.
        :param possible_next_actions: See subclass' `get_max_q_values` documentation.
        """
        raise NotImplementedError()

    def get_q_values(
        self,
        states: str,
        actions: str,
        use_target_network: bool,
    ) -> str:
        """
        Takes in a set of next_states and corresponding next_actions. For each
        (next_state_i, next_action_i) pair, calculates Q(next_state, next_action).
        Returns these q values in a Numpy array of shape (batch_size, 1).

        :param next_states: Numpy array with shape (batch_size, state_dim). The
            ith row is a representation of the ith transition's next_state.
        :param next_actions: See subclass' `get_sarsa_values` documentation.
        """
        raise NotImplementedError()

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
            Comptutes Loss of Q(states, actions) with respect to q_vals_targets.
            Updates Q Network's weights according to loss and optimizer.

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row is a representation of the ith transition's action.
        :param q_vals_targets: Numpy array with shape (batch_size, 1). The ith
            row is the label to train against for the data from the ith transition.
        """
        raise NotImplementedError()

    def _create_reward_train_net(self) -> None:
        raise NotImplementedError()

    def _create_rl_train_net(self) -> None:
        raise NotImplementedError()

    def _create_q_score_net(self) -> None:
        self.q_score_model = ModelHelper(name="q_score_" + self.model_id)
        C2.set_model(self.q_score_model)
        self.q_score_output = self.get_q_values('states', 'actions', True)
        workspace.RunNetOnce(self.q_score_model.param_init_net)
        workspace.CreateNet(self.q_score_model.net)
        C2.set_model(None)

    def train_numpy(
        self,
        tdp: TrainingDataPage,
        evaluator: Optional[Evaluator],
    ):
        workspace.FeedBlob('states', tdp.states)
        workspace.FeedBlob('actions', tdp.actions)
        workspace.FeedBlob('rewards', tdp.rewards)
        workspace.FeedBlob('next_states', tdp.next_states)
        workspace.FeedBlob('not_terminals', tdp.not_terminals)
        if self.maxq_learning:
            if isinstance(tdp.possible_next_actions, StackedArray):
                workspace.FeedBlob(
                    'possible_next_actions', tdp.possible_next_actions.values
                )
                workspace.FeedBlob(
                    'possible_next_actions_lengths',
                    tdp.possible_next_actions.lengths
                )
            else:
                workspace.FeedBlob(
                    'possible_next_actions', tdp.possible_next_actions
                )
        else:
            workspace.FeedBlob('next_actions', tdp.next_actions)
        self.train(tdp.reward_timelines, evaluator)

    def train(
        self,
        reward_timelines: Optional[List[Dict[int, float]]],
        evaluator: Optional[Evaluator],
    ) -> None:
        assert self.rl_train_model is not None
        assert self.reward_train_model is not None
        assert self.q_score_model is not None

        if self.training_iteration >= self.reward_burnin:
            if self.training_iteration == self.reward_burnin:
                logger.info(
                    "Minibatch number == reward_burnin. Starting RL updates."
                )
                self.target_network.enable_slow_updates()
            workspace.RunNet(self.rl_train_model.net)
        else:
            workspace.RunNet(self.reward_train_model.net)

        self.target_network.target_update()
        self.training_iteration += 1
        workspace.RunNet(self.q_score_model.net)
        if evaluator is not None:
            assert reward_timelines is not None
            assert self.loss_blob is not None
            evaluator.report(
                reward_timelines,
                workspace.FetchBlob(self.q_score_output),
                workspace.FetchBlob(self.loss_blob),
            )
