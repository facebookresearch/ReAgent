#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
import torch
from ml.rl.caffe_utils import softmax
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


logger = logging.getLogger(__name__)


class GymPredictor(object):
    def __init__(self, trainer, action_dim):
        self.trainer = trainer
        self.action_dim = action_dim

    def policy(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def estimate_reward(self):
        raise NotImplementedError()


class GymDQNPredictor(GymPredictor):
    def policy(self, states):
        if isinstance(self.trainer, DQNTrainer):
            input = [states]
        elif isinstance(self.trainer, ParametricDQNTrainer):
            num_actions = self.action_dim
            actions = np.eye(num_actions, dtype=np.float32)
            actions = np.tile(actions, reps=(len(states), 1))
            states = np.repeat(states, repeats=num_actions, axis=0)
            input = (states, actions)
        else:
            raise NotImplementedError("Invalid trainer passed to GymPredictor")
        q_scores = self.trainer.internal_prediction(*input)
        if isinstance(self.trainer, DQNTrainer):
            assert q_scores.shape[0] == 1
            q_scores = q_scores[0]
        q_scores_softmax = softmax(
            torch.from_numpy(q_scores.reshape(1, -1)), self.trainer.rl_temperature
        ).numpy()[0]
        if np.isnan(q_scores_softmax).any() or np.max(q_scores_softmax) < 1e-3:
            q_scores_softmax[:] = 1.0 / q_scores_softmax.shape[0]
        policies = [
            np.argmax(q_scores),
            np.random.choice(q_scores.shape[0], p=q_scores_softmax),
        ]
        return policies

    def predict(self, states):
        if isinstance(self.trainer, DQNTrainer):
            input = [states]
        elif isinstance(self.trainer, ParametricDQNTrainer):
            num_actions = self.action_dim
            actions = np.eye(num_actions, dtype=np.float32)
            actions = np.tile(actions, reps=(len(states), 1))
            states = np.repeat(states, repeats=num_actions, axis=0)
            input = (states, actions)
        else:
            raise NotImplementedError("Invalid trainer passed to GymPredictor")
        q_scores = self.trainer.internal_prediction(*input)
        return q_scores

    def estimate_reward(self, states):
        if isinstance(self.trainer, DQNTrainer):
            input = [states]
        elif isinstance(self.trainer, ParametricDQNTrainer):
            num_actions = self.action_dim
            actions = np.eye(num_actions, dtype=np.float32)
            actions = np.tile(actions, reps=(len(states), 1))
            states = np.repeat(states, repeats=num_actions, axis=0)
            input = (states, actions)
        else:
            raise NotImplementedError("Invalid trainer passed to GymPredictor")
        reward_estimates = self.trainer.internal_reward_estimation(*input)
        return reward_estimates


class GymDDPGPredictor(GymPredictor):
    def policy(self, states, add_action_noise=False):
        return self.trainer.internal_prediction(states, add_action_noise)


class GymSACPredictor(GymPredictor):
    def policy(self, states):
        actions = self.trainer.internal_prediction(states)
        actions = actions.data.numpy()
        return actions
