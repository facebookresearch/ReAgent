#!/usr/bin/env python3

from collections import Counter
from typing import List
import numpy as np

import logging

logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, action_names, evaluator_batch_size) -> None:
        self.action_names = action_names
        self.mc_loss: List[float] = []
        self.td_loss: List[float] = []
        self.value_inverse_propensity_score: List[float] = []
        self.value_direct_method: List[float] = []
        self.value_doubly_robust: List[float] = []
        self.reward_inverse_propensity_score: List[float] = []
        self.reward_direct_method: List[float] = []
        self.reward_doubly_robust: List[float] = []

        self.evaluator_batch_size = evaluator_batch_size

        self.td_loss_batches: List[np.ndarray] = []
        self.logged_actions_batches: List[np.ndarray] = []
        self.logged_propensities_batches: List[np.ndarray] = []
        self.logged_rewards_batches: List[np.ndarray] = []
        self.logged_values_batches: List[np.ndarray] = []
        self.model_propensities_batches: List[np.ndarray] = []
        self.model_values_batches: List[np.ndarray] = []
        self.model_values_on_logged_actions_batches: List[np.ndarray] = []
        self.model_action_idxs_batches: List[np.ndarray] = []

        self.all_batches = [
            self.td_loss_batches,
            self.logged_actions_batches,
            self.logged_propensities_batches,
            self.logged_rewards_batches,
            self.logged_values_batches,
            self.model_propensities_batches,
            self.model_values_batches,
            self.model_values_on_logged_actions_batches,
            self.model_action_idxs_batches,
        ]

    def report(
        self,
        td_loss,
        logged_actions,
        logged_propensities,
        logged_rewards,
        logged_values,
        model_propensities,
        model_values,
        model_values_on_logged_actions,
        model_action_idxs,
    ):
        input_list = [
            td_loss,
            logged_actions,
            logged_propensities,
            logged_rewards,
            logged_values,
            model_propensities,
            model_values,
            model_values_on_logged_actions,
            model_action_idxs,
        ]
        for i, input in enumerate(input_list):
            if input is None:
                assert (
                    len(self.all_batches[i]) == 0
                ), "Missing a batch.  Either omit completely or fill every time"
            else:
                self.all_batches[i].append(input)

        if len(self.td_loss_batches) >= self.evaluator_batch_size:
            self.evaluate_batch()
            self.clear_evaluation_containers()

    def clear_evaluation_containers(self):
        for batch in self.all_batches:
            batch.clear()

    def evaluate_batch(self):
        merged_inputs = []
        for batch in self.all_batches:
            if len(batch) > 0:
                merged_inputs.append(np.vstack(batch))
            else:
                merged_inputs.append(None)
        td_loss, logged_actions, logged_propensities, logged_rewards, logged_values, model_propensities, model_values, model_values_on_logged_actions, model_action_idxs = (
            merged_inputs
        )

        logger.info("Evaluating on {} batches".format(len(self.td_loss_batches)))
        print_details = "Evaluator:\n"
        if td_loss is not None:
            td_loss_mean = float(np.mean(td_loss))
            self.td_loss.append(td_loss_mean)
            print_details = print_details + "TD LOSS: {0:.3f}\n".format(td_loss_mean)
        if logged_values is not None:
            mc_loss = float(
                np.mean(np.abs(logged_values - model_values_on_logged_actions))
            )
            self.mc_loss.append(mc_loss)
            print_details = print_details + "MC LOSS: {0:.3f}\n".format(mc_loss)

        if (
            logged_actions is not None
            and model_propensities is not None
            and model_values is not None
        ):
            if logged_propensities is None:
                # Assume a deterministic model
                logged_propensities = np.ones(logged_values.shape)

            v_ips, v_dm, v_dr = self.doubly_robust_policy_estimation(
                logged_actions,
                logged_values,
                logged_propensities,
                model_propensities,
                model_values,
            )
            self.value_inverse_propensity_score.append(v_ips)
            self.value_direct_method.append(v_dm)
            self.value_doubly_robust.append(v_dr)

            print_details += "Value Inverse Propensity Score : {0:.3f}\n".format(v_ips)
            print_details += "Value Direct Method            : {0:.3f}\n".format(v_dm)
            print_details += "Value Doubly Robust P.E.       : {0:.3f}\n".format(v_dr)

            r_ips, r_dm, r_dr = self.doubly_robust_policy_estimation(
                logged_actions,
                logged_rewards,
                logged_propensities,
                model_propensities,
                None,
            )
            self.reward_inverse_propensity_score.append(r_ips)
            self.reward_direct_method.append(r_dm)
            self.reward_doubly_robust.append(r_dr)

            print_details += "Reward Inverse Propensity Score : {0:.3f}\n".format(r_ips)
            print_details += "Reward Direct Method            : {0:.3f}\n".format(r_dm)
            print_details += "Reward Doubly Robust P.E.       : {0:.3f}\n".format(r_dr)

        if logged_actions is not None and model_action_idxs is not None:
            logged_action_counter = Counter(np.argmax(logged_actions, axis=1))
            model_action_counter = Counter(model_action_idxs.reshape(-1))
            print_details += "The distribution of logged actions : {}\n".format(
                {
                    action_name: logged_action_counter[i]
                    for i, action_name in enumerate(self.action_names)
                }
            )
            print_details += "The distribution of model actions : {}\n".format(
                {
                    action_name: model_action_counter[i]
                    for i, action_name in enumerate(self.action_names)
                }
            )

        print_details += "Evaluator Finished"
        for print_detail in print_details.split("\n"):
            logger.info(print_detail)

    def get_recent_td_loss(self):
        begin = max(0, len(self.td_loss) - 100)
        return np.mean(np.array(self.td_loss[begin:]))

    def get_recent_mc_loss(self):
        begin = max(0, len(self.mc_loss) - 100)
        return np.mean(np.array(self.mc_loss[begin:]))

    def get_recent_inverse_propensity_score(self):
        begin = max(0, len(self.reward_inverse_propensity_score) - 100)
        return np.mean(np.array(self.reward_inverse_propensity_score[begin:]))

    def get_recent_direct_method(self):
        begin = max(0, len(self.reward_direct_method) - 100)
        return np.mean(np.array(self.reward_direct_method[begin:]))

    def get_recent_doubly_robust(self):
        begin = max(0, len(self.reward_doubly_robust) - 100)
        return np.mean(np.array(self.reward_doubly_robust[begin:]))

    def doubly_robust_policy_estimation(
        self,
        logged_actions,
        logged_rewards,
        logged_propensities,
        target_propensities,
        estimated_values,
    ):
        # For details, visit https://arxiv.org/pdf/1612.01205.pdf
        num_examples = len(logged_actions)

        if estimated_values is None:
            # Fill with zero, equivalent to just doing IPS
            estimated_values = np.zeros(target_propensities.shape)
            direct_method_values = np.zeros([num_examples, 1], dtype=np.float32)
        else:
            direct_method_values = np.sum(
                target_propensities * estimated_values, axis=1, keepdims=True
            )

        total_reward = np.sum(logged_rewards)

        target_propensity_for_action = np.sum(
            target_propensities * logged_actions, axis=1, keepdims=True
        )
        importance_weight = target_propensity_for_action / logged_propensities
        ips = importance_weight * logged_rewards
        estimated_values_for_action = np.sum(
            estimated_values * logged_actions, axis=1, keepdims=True
        )
        doubly_robust = (
            importance_weight * (logged_rewards - estimated_values_for_action)
        ) + direct_method_values

        return (
            float(np.sum(ips) / total_reward),
            float(np.sum(direct_method_values) / total_reward),
            float(np.sum(doubly_robust) / total_reward),
        )

    @staticmethod
    def softmax(x, temperature):
        """Compute softmax values for each sets of scores in x."""
        x = x / temperature
        x -= np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)
