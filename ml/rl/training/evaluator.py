#!/usr/bin/env python3


import numpy as np

import logging

logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self):
        self.mc_loss = []
        self.td_loss = []
        self.value_inverse_propensity_score = []
        self.value_direct_method = []
        self.value_doubly_robust = []
        self.reward_inverse_propensity_score = []
        self.reward_direct_method = []
        self.reward_doubly_robust = []

    def report(self, episode_values, predictions, td_loss):
        print_details = ""
        if episode_values is not None:
            mc_loss = float(np.mean(np.abs(episode_values - predictions)))
            self.mc_loss.append(mc_loss)
            print_details = print_details + "MC LOSS: {0:.3f} ".format(mc_loss)
        if td_loss is not None:
            td_loss_mean = float(np.mean(td_loss))
            self.td_loss.append(td_loss_mean)
            print_details = print_details + "TD LOSS: {0:.3f} ".format(td_loss_mean)
        logger.info(print_details)

    def get_recent_td_loss(self):
        begin = max(0, len(self.td_loss) - 100)
        return np.mean(np.array(self.td_loss[begin:]))

    def get_recent_mc_loss(self):
        begin = max(0, len(self.mc_loss) - 100)
        return np.mean(np.array(self.mc_loss[begin:]))

    def doubly_robust_policy_estimation(
        self,
        num_actions,
        logged_actions,
        logged_values,
        logged_propensities,
        target_propensities,
        estimated_values,
    ):
        # For details, visit https://arxiv.org/pdf/1612.01205.pdf
        num_examples = len(logged_actions)

        direct_method_values = np.zeros(num_examples, dtype=np.float32)
        for x in range(num_examples):
            for action in range(num_actions):
                direct_method_values[x] += (
                    target_propensities[x][action] * estimated_values[x][action]
                )

        ips_sum = 0.0
        direct_method_sum = np.sum(direct_method_values)
        doubly_robust_sum = 0.0
        for x in range(num_examples):
            logged_value = logged_values[x]
            direct_method_value = direct_method_values[x]
            logged_propensity = logged_propensities[x]
            target_propensity = target_propensities[x][logged_actions[x]]

            if logged_propensity > 1e-6:
                importance_weight = target_propensity / logged_propensity
            else:
                importance_weight = 0
            ips_sum += importance_weight * logged_value

            doubly_robust_sum += (
                importance_weight
                * (logged_value - estimated_values[x][logged_actions[x]])
            ) + direct_method_value

        return (
            ips_sum / float(num_examples),
            direct_method_sum / float(num_examples),
            doubly_robust_sum / float(num_examples),
        )

    @staticmethod
    def softmax(x, temperature):
        """Compute softmax values for each sets of scores in x."""
        x = x / temperature
        x -= np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)
