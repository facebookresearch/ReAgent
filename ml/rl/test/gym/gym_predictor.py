#!/usr/bin/env python3

import numpy as np

from caffe2.python import core, workspace

from ml.rl.training.continuous_action_dqn_trainer import ContinuousActionDQNTrainer
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer
from ml.rl.training.evaluator import Evaluator

import logging

logger = logging.getLogger(__name__)


class GymPredictor(object):
    def __init__(self, trainer, c2_device=None):
        self.c2_device = c2_device
        self.trainer = trainer

    def policy(self):
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
                states = np.tile(states, (num_actions, 1))
                workspace.FeedBlob("states", states)
                actions = np.eye(num_actions, dtype=np.float32)
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


class GymDQNPredictorPytorch(GymPredictor):
    def __init__(self, trainer):
        GymPredictor.__init__(self, trainer)

    def policy(self, states):
        q_scores = self.trainer.internal_prediction(states)
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


class GymDDPGPredictor(GymPredictor):
    def __init__(self, trainer):
        GymPredictor.__init__(self, trainer)

    def policy(self, states, add_action_noise=False):
        return self.trainer.internal_prediction(states, add_action_noise)
