#!/usr/bin/env python3

import numpy as np

from caffe2.python import core, workspace

from ml.rl.training.continuous_action_dqn_trainer import ContinuousActionDQNTrainer
from ml.rl.training.discrete_action_trainer import DiscreteActionTrainer

import logging
logger = logging.getLogger(__name__)


class GymPredictor(object):
    def __init__(self, c2_device, trainer):
        self.c2_device = c2_device
        self.trainer = trainer

    def policy(self, states):
        with core.DeviceScope(self.c2_device):
            if isinstance(self.trainer, DiscreteActionTrainer):
                workspace.FeedBlob('states', states)
            elif isinstance(self.trainer, ContinuousActionDQNTrainer):
                num_actions = len(self.trainer.action_normalization_parameters)
                states = np.tile(states, (num_actions, 1))
                workspace.FeedBlob('states', states)
                actions = np.eye(num_actions, dtype=np.float32)
                workspace.FeedBlob('actions', actions)
            else:
                raise NotImplementedError(
                    "Invalid trainer passed to GymPredictor"
                )
            workspace.RunNetOnce(self.trainer.internal_policy_model.net)
            policy_output_blob = self.trainer.internal_policy_output
            q_scores = workspace.FetchBlob(policy_output_blob)
            if isinstance(self.trainer, DiscreteActionTrainer):
                assert q_scores.shape[0] == 1
                q_scores = q_scores[0]
            q_scores_softmax = GymPredictor._softmax(
                q_scores, self.trainer.rl_temperature
            )
            if np.isnan(q_scores_softmax).any() or \
                    np.max(q_scores_softmax) < 1e-3:
                q_scores_softmax[:] = 1.0 / q_scores_softmax.shape[0]
            policies = [
                np.argmax(q_scores),
                np.random.choice(q_scores.shape[0], p=q_scores_softmax),
            ]
            return policies

    @staticmethod
    def _softmax(x, temperature):
        """Compute softmax values for each sets of scores in x."""
        x = x / temperature
        x -= np.max(x)
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)
