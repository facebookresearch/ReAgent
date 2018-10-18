#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
from caffe2.python import core, workspace
from ml.rl.training.continuous_action_dqn_trainer import ContinuousActionDQNTrainer
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


logger = logging.getLogger(__name__)


class GymPredictor(object):
    def __init__(self, trainer, c2_device=None):
        self.c2_device = c2_device
        self.trainer = trainer

    def policy(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def estimate_reward(self):
        raise NotImplementedError()


class GymDQNPredictor(GymPredictor):
    def __init__(self, trainer, c2_device):
        GymPredictor.__init__(self, trainer, c2_device)

    def policy(self, states):
        with core.DeviceScope(self.c2_device):
            if isinstance(self.trainer, DiscreteActionTrainer):
                workspace.FeedBlob("states", states)
            elif isinstance(self.trainer, ContinuousActionDQNTrainer):
                num_actions = len(self.trainer.action_normalization_parameters)
                actions = np.eye(num_actions, dtype=np.float32)
                actions = np.tile(actions, reps=(len(states), 1))
                states = np.repeat(states, repeats=num_actions, axis=0)
                workspace.FeedBlob("states", states)
                workspace.FeedBlob("actions", actions)
            else:
                raise NotImplementedError("Invalid trainer passed to GymPredictor")
            workspace.RunNetOnce(self.trainer.internal_policy_model.net)
            policy_output_blob = self.trainer.internal_policy_output
            q_scores = workspace.FetchBlob(policy_output_blob)
            if isinstance(self.trainer, DiscreteActionTrainer):
                assert q_scores.shape[0] == 1
                q_scores = q_scores[0]
            q_scores_softmax = Evaluator.softmax(
                q_scores.reshape(1, -1), self.trainer.rl_temperature
            )[0]
            if np.isnan(q_scores_softmax).any() or np.max(q_scores_softmax) < 1e-3:
                q_scores_softmax[:] = 1.0 / q_scores_softmax.shape[0]
            policies = [
                np.argmax(q_scores),
                np.random.choice(q_scores.shape[0], p=q_scores_softmax),
            ]
            return policies

    def predict(self, states):
        with core.DeviceScope(self.c2_device):
            if isinstance(self.trainer, DiscreteActionTrainer):
                workspace.FeedBlob("states", states)
            elif isinstance(self.trainer, ContinuousActionDQNTrainer):
                num_actions = len(self.trainer.action_normalization_parameters)
                actions = np.eye(num_actions, dtype=np.float32)
                actions = np.tile(actions, reps=(len(states), 1))
                states = np.repeat(states, repeats=num_actions, axis=0)
                workspace.FeedBlob("states", states)
                workspace.FeedBlob("actions", actions)
            else:
                raise NotImplementedError("Invalid trainer passed to GymPredictor")
            workspace.RunNetOnce(self.trainer.internal_policy_model.net)
            policy_output_blob = self.trainer.internal_policy_output
            print(self.trainer.internal_policy_output)
            q_scores = workspace.FetchBlob(policy_output_blob)
            return q_scores


class GymDQNPredictorPytorch(GymPredictor):
    def __init__(self, trainer):
        GymPredictor.__init__(self, trainer)

    def policy(self, states):
        if isinstance(self.trainer, DQNTrainer):
            input = states
        elif isinstance(self.trainer, ParametricDQNTrainer):
            num_actions = len(self.trainer.action_normalization_parameters)
            actions = np.eye(num_actions, dtype=np.float32)
            actions = np.tile(actions, reps=(len(states), 1))
            states = np.repeat(states, repeats=num_actions, axis=0)
            input = np.hstack((states, actions))
        else:
            raise NotImplementedError("Invalid trainer passed to GymPredictor")
        q_scores = self.trainer.internal_prediction(input)
        if isinstance(self.trainer, DQNTrainer):
            assert q_scores.shape[0] == 1
            q_scores = q_scores[0]
        q_scores_softmax = Evaluator.softmax(
            q_scores.reshape(1, -1), self.trainer.rl_temperature
        )[0]
        if np.isnan(q_scores_softmax).any() or np.max(q_scores_softmax) < 1e-3:
            q_scores_softmax[:] = 1.0 / q_scores_softmax.shape[0]
        policies = [
            np.argmax(q_scores),
            np.random.choice(q_scores.shape[0], p=q_scores_softmax),
        ]
        return policies

    def predict(self, states):
        if isinstance(self.trainer, DQNTrainer):
            input = states
        elif isinstance(self.trainer, ParametricDQNTrainer):
            num_actions = len(self.trainer.action_normalization_parameters)
            actions = np.eye(num_actions, dtype=np.float32)
            actions = np.tile(actions, reps=(len(states), 1))
            states = np.repeat(states, repeats=num_actions, axis=0)
            input = np.hstack((states, actions))
        else:
            raise NotImplementedError("Invalid trainer passed to GymPredictor")
        q_scores = self.trainer.internal_prediction(input)
        return q_scores

    def estimate_reward(self, states):
        if isinstance(self.trainer, DQNTrainer):
            input = states
        elif isinstance(self.trainer, ParametricDQNTrainer):
            num_actions = len(self.trainer.action_normalization_parameters)
            actions = np.eye(num_actions, dtype=np.float32)
            actions = np.tile(actions, reps=(len(states), 1))
            states = np.repeat(states, repeats=num_actions, axis=0)
            input = np.hstack((states, actions))
        else:
            raise NotImplementedError("Invalid trainer passed to GymPredictor")
        reward_estimates = self.trainer.internal_reward_estimation(input)
        return reward_estimates


class GymDDPGPredictor(GymPredictor):
    def __init__(self, trainer):
        GymPredictor.__init__(self, trainer)

    def policy(self, states, add_action_noise=False):
        return self.trainer.internal_prediction(states, add_action_noise)
