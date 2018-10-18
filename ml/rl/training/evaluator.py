#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import hashlib
import itertools
import logging
import math
from collections import Counter
from typing import List, NamedTuple, Optional

import numpy as np
import scipy as sp
import torch
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class CPE_Estimate(NamedTuple):
    normalized: float
    raw: float


class DiscreteActionSample:
    __slots__ = [
        "mdp_id",
        "sequence_number",
        "state",
        "action",
        "reward",
        "propensity",
        "terminal",
    ]

    def __init__(
        self, mdp_id, sequence_number, state, action, reward, propensity, terminal
    ):
        self.mdp_id = mdp_id
        self.sequence_number = sequence_number
        self.state = state
        self.action = action
        self.reward = reward
        self.propensity = propensity
        self.terminal = terminal


class ParametricActionSample:
    __slots__ = [
        "mdp_id",
        "sequence_number",
        "logged_state_action",
        "reward",
        "propensity",
        "terminal",
        "possible_state_actions",
    ]

    def __init__(
        self,
        mdp_id,
        sequence_number,
        logged_state_action,
        reward,
        propensity,
        terminal,
        possible_state_actions,
    ):
        self.mdp_id = mdp_id
        self.sequence_number = sequence_number
        self.logged_state_action = logged_state_action
        self.reward = reward
        self.propensity = propensity
        self.terminal = terminal
        self.possible_state_actions = possible_state_actions


class Evaluator(object):
    NUM_J_STEPS_FOR_MAGIC_ESTIMATOR = 25
    NUM_SUBSETS_FOR_CB_ESTIMATES = 25
    CONFIDENCE_INTERVAL = 0.9
    RECENT_WINDOW_SIZE = 100

    def __init__(
        self, action_names, evaluator_batch_size, gamma, model, mdp_sampled_rate
    ) -> None:
        self.action_names = action_names
        self.mc_loss: List[float] = []
        self.td_loss: List[float] = []
        self.reward_loss: List[float] = []
        self.value_inverse_propensity_score: List[CPE_Estimate] = []
        self.value_direct_method: List[CPE_Estimate] = []
        self.value_doubly_robust: List[CPE_Estimate] = []
        self.value_sequential_doubly_robust: List[CPE_Estimate] = []
        self.value_weighted_doubly_robust: List[CPE_Estimate] = []
        self.value_magic_doubly_robust: List[CPE_Estimate] = []
        self.true_value_PE: List[float] = []
        self.true_discounted_value_PE: List[float] = []
        self.reward_inverse_propensity_score: List[CPE_Estimate] = []
        self.reward_direct_method: List[CPE_Estimate] = []
        self.reward_doubly_robust: List[CPE_Estimate] = []

        self.evaluator_batch_size = evaluator_batch_size

        self.td_loss_batches: List[np.ndarray] = []
        self.logged_actions_batches: List[np.ndarray] = []
        self.logged_propensities_batches: List[np.ndarray] = []
        self.logged_rewards_batches: List[np.ndarray] = []
        self.logged_values_batches: List[np.ndarray] = []
        self.model_propensities_batches: List[np.ndarray] = []
        self.model_rewards_batches: List[np.ndaary] = []
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
            self.model_rewards_batches,
            self.model_values_batches,
            self.model_values_on_logged_actions_batches,
            self.model_action_idxs_batches,
        ]

        self.gamma = gamma
        self.model = model
        self.mdp_sampled_rate = mdp_sampled_rate

        self.hash_func = hashlib.sha512
        self.max_hash = 2 ** (self.hash_func().digest_size * 8)

        self.unshuffled_samples: List = []
        self.unshuffled_actions = np.array([])
        self.unshuffled_rewards = np.array([])
        self.unshuffled_logged_propensities = np.array([])
        self.unshuffled_terminals = np.array([])
        self.unshuffled_target_propensities = np.array([])
        self.unshuffled_estimated_q_values = np.array([])

    def report(
        self,
        td_loss,
        logged_actions,
        logged_propensities,
        logged_rewards,
        logged_values,
        model_propensities,
        model_rewards,
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
            model_rewards,
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
        td_loss, logged_actions, logged_propensities, logged_rewards, logged_values, model_propensities, model_rewards, model_values, model_values_on_logged_actions, model_action_idxs = (
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
            and model_rewards is not None
        ):
            if logged_propensities is None:
                # Assume a deterministic model
                logged_propensities = np.ones(logged_values.shape)

            r_ips, r_dm, r_dr = self.doubly_robust_one_step_policy_estimation(
                logged_actions,
                logged_rewards,
                logged_propensities,
                model_propensities,
                model_rewards,
            )
            self.reward_inverse_propensity_score.append(r_ips)
            self.reward_direct_method.append(r_dm)
            self.reward_doubly_robust.append(r_dr)

            print_details += "Reward Inverse Propensity Score : normalized {0:.3f} raw {1:.3f}\n".format(
                r_ips.normalized, r_ips.raw
            )
            print_details += "Reward Direct Method            : normalized {0:.3f} raw {1:.3f}\n".format(
                r_dm.normalized, r_dm.raw
            )
            print_details += "Reward Doubly Robust P.E.       : normalized {0:.3f} raw {1:.3f}\n".format(
                r_dr.normalized, r_dr.raw
            )

            v_ips, v_dm, v_dr = self.doubly_robust_one_step_policy_estimation(
                logged_actions,
                logged_values,
                logged_propensities,
                model_propensities,
                model_values,
            )
            self.value_inverse_propensity_score.append(v_ips)
            self.value_direct_method.append(v_dm)
            self.value_doubly_robust.append(v_dr)

            print_details += "Value Inverse Propensity Score    : normalized {0:.3f} raw {1:.3f}\n".format(
                v_ips.normalized, v_ips.raw
            )
            print_details += "Value Direct Method               : normalized {0:.3f} raw {1:.3f}\n".format(
                v_dm.normalized, v_dm.raw
            )
            print_details += "Value One-Step Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}\n".format(
                v_dr.normalized, v_dr.raw
            )

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

        print_details += "Batch Evaluator Finished"
        for print_detail in print_details.split("\n"):
            logger.info(print_detail)

    def mdp_id_to_probability(self, mdp_id):
        """
        Return a reproducible random float in the interval [0, 1) for the mdp_id.
        """
        hash_digest = self.hash_func(mdp_id).digest()
        hash_int = int.from_bytes(hash_digest, "big")
        return hash_int / self.max_hash

    def collect_discrete_action_samples(
        self,
        mdp_ids,
        sequence_numbers,
        states,
        logged_actions,
        logged_rewards,
        logged_propensities,
        logged_terminals,
    ):
        for i in range(len(mdp_ids)):
            mdp_id = mdp_ids[i]
            if self.mdp_id_to_probability(mdp_id) < self.mdp_sampled_rate:
                self.unshuffled_samples.append(
                    DiscreteActionSample(
                        mdp_id=mdp_id,
                        sequence_number=sequence_numbers[i],
                        state=states[i],
                        action=logged_actions[i],
                        reward=logged_rewards[i],
                        propensity=logged_propensities[i],
                        terminal=logged_terminals[i],
                    )
                )

    def collect_parametric_action_samples(
        self,
        mdp_ids,
        sequence_numbers,
        logged_state_actions,
        logged_rewards,
        logged_propensities,
        logged_terminals,
        possible_state_actions,
        pas_lens,
    ):
        cum_pas_lens = np.zeros(len(pas_lens) + 1, dtype=np.int64)
        cum_pas_lens[1:] = np.cumsum(pas_lens)

        for i in range(len(mdp_ids)):
            mdp_id = mdp_ids[i]
            if self.mdp_id_to_probability(mdp_id) < self.mdp_sampled_rate:
                self.unshuffled_samples.append(
                    ParametricActionSample(
                        mdp_id=mdp_id,
                        sequence_number=sequence_numbers[i],
                        logged_state_action=logged_state_actions[i],
                        reward=logged_rewards[i],
                        propensity=logged_propensities[i],
                        terminal=logged_terminals[i],
                        possible_state_actions=possible_state_actions[
                            cum_pas_lens[i] : cum_pas_lens[i + 1]
                        ],
                    )
                )

    def recover_samples_to_be_unshuffled(self):
        if len(self.unshuffled_samples) == 0:
            return

        # sort to recover unshuffled samples
        self.unshuffled_samples.sort(key=lambda x: (x.mdp_id, x.sequence_number))

        # add terminal at the end of every mdp sequence
        for i in range(len(self.unshuffled_samples) - 1):
            if (
                self.unshuffled_samples[i].mdp_id
                != self.unshuffled_samples[i + 1].mdp_id
            ):
                self.unshuffled_samples[i].terminal = True
        self.unshuffled_samples[-1].terminal = True

        self.unshuffled_rewards = np.array(
            [x.reward for x in self.unshuffled_samples]
        ).reshape(-1, 1)
        self.unshuffled_logged_propensities = np.array(
            [x.propensity for x in self.unshuffled_samples]
        ).reshape(-1, 1)
        self.unshuffled_terminals = (
            np.array([x.terminal for x in self.unshuffled_samples])
            .astype(np.bool)
            .reshape(-1, 1)
        )

        if isinstance(self.unshuffled_samples[0], DiscreteActionSample):
            unshuffled_states = torch.tensor(
                np.array([x.state for x in self.unshuffled_samples]),
                dtype=torch.float32,
            ).type(self.model.dtype)
            self.unshuffled_estimated_q_values = (
                self.model.calculate_q_values(unshuffled_states).cpu().numpy()
            )
            self.unshuffled_actions = np.array(
                [x.action for x in self.unshuffled_samples]
            )
        else:
            unshuffled_possible_state_actions = torch.tensor(
                np.array(
                    [
                        sa
                        for x in self.unshuffled_samples
                        for sa in x.possible_state_actions
                    ]
                )
            ).type(self.model.dtype)
            pas_lens = torch.tensor(
                np.array(
                    [len(x.possible_state_actions) for x in self.unshuffled_samples]
                ),
                dtype=torch.int64,
            ).type(self.model.dtypelong)
            self.unshuffled_estimated_q_values = (
                self.model.calculate_q_values(
                    unshuffled_possible_state_actions, pas_lens
                )
                .cpu()
                .numpy()
            )
            unshuffled_logged_state_actions = torch.tensor(
                np.array([x.logged_state_action for x in self.unshuffled_samples]),
                dtype=torch.float32,
            ).type(self.model.dtype)
            q_value_for_logged_state_action = (
                self.model.calculate_q_values(
                    unshuffled_logged_state_actions,
                    torch.ones([unshuffled_logged_state_actions.shape[0]]).type(
                        self.model.dtypelong
                    ),
                )
                .cpu()
                .numpy()
            )
            self.unshuffled_actions = (
                q_value_for_logged_state_action == self.unshuffled_estimated_q_values
            ).astype(np.int64)

        self.unshuffled_target_propensities = Evaluator.softmax(
            self.unshuffled_estimated_q_values, self.model.rl_temperature
        )

    def clear_collected_samples(self):
        self.unshuffled_samples.clear()
        self.unshuffled_actions = np.array([])
        self.unshuffled_rewards = np.array([])
        self.unshuffled_logged_propensities = np.array([])
        self.unshuffled_terminals = np.array([])
        self.unshuffled_target_propensities = np.array([])
        self.unshuffled_estimated_q_values = np.array([])

    def score_cpe(self):
        if len(self.unshuffled_samples) == 0 or len(self.unshuffled_actions) == 0:
            return

        logger.info(
            "CPE Evaluator after {} epoches".format(
                len(self.value_sequential_doubly_robust) + 1
            )
        )

        assert len(self.unshuffled_actions.shape) == 2, "Invalid number of dimensions"
        assert len(self.unshuffled_rewards.shape) == 2, "Invalid number of dimensions"
        assert len(self.unshuffled_terminals.shape) == 2, "Invalid number of dimensions"
        assert (
            len(self.unshuffled_logged_propensities.shape) == 2
        ), "Invalid number of dimensions"
        assert (
            len(self.unshuffled_target_propensities.shape) == 2
        ), "Invalid number of dimensions"
        assert (
            len(self.unshuffled_estimated_q_values.shape) == 2
        ), "Invalid number of dimensions"
        assert (
            len(self.unshuffled_logged_propensities.shape) == 2
        ), "Invalid number of dimensions"

        assert (
            self.unshuffled_actions.shape[0] == self.unshuffled_rewards.shape[0]
        ), "Mismatched minibatch size"
        assert (
            self.unshuffled_actions.shape[0] == self.unshuffled_terminals.shape[0]
        ), "Mismatched minibatch size"
        assert (
            self.unshuffled_actions.shape[0]
            == self.unshuffled_logged_propensities.shape[0]
        ), "Mismatched minibatch size"
        assert (
            self.unshuffled_actions.shape[0]
            == self.unshuffled_target_propensities.shape[0]
        ), "Mismatched minibatch size"
        assert (
            self.unshuffled_actions.shape[0]
            == self.unshuffled_estimated_q_values.shape[0]
        ), "Mismatched minibatch size"

        # [N,1]
        assert self.unshuffled_rewards.shape[1] == 1
        assert self.unshuffled_terminals.shape[1] == 1
        assert self.unshuffled_logged_propensities.shape[1] == 1

        # [N, Max_Num_Possible_actions]
        assert (
            self.unshuffled_target_propensities.shape[1]
            == self.unshuffled_estimated_q_values.shape[1]
        )
        assert (
            self.unshuffled_target_propensities.shape[1]
            == self.unshuffled_actions.shape[1]
        )

        sequential_doubly_robust = self.doubly_robust_sequential_policy_estimation(
            self.unshuffled_actions,
            self.unshuffled_rewards,
            self.unshuffled_terminals,
            self.unshuffled_logged_propensities,
            self.unshuffled_target_propensities,
            self.unshuffled_estimated_q_values,
        )
        self.value_sequential_doubly_robust.append(sequential_doubly_robust)

        logger.info(
            "Value Sequential Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}".format(
                sequential_doubly_robust.normalized, sequential_doubly_robust.raw
            )
        )

        weighted_doubly_robust = self.weighted_doubly_robust_sequential_policy_estimation(
            self.unshuffled_actions,
            self.unshuffled_rewards,
            self.unshuffled_terminals,
            self.unshuffled_logged_propensities,
            self.unshuffled_target_propensities,
            self.unshuffled_estimated_q_values,
            num_j_steps=1,
            whether_self_normalize_importance_weights=True,
        )
        self.value_weighted_doubly_robust.append(weighted_doubly_robust)

        logger.info(
            "Value Weighted Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}".format(
                weighted_doubly_robust.normalized, weighted_doubly_robust.raw
            )
        )

        magic_doubly_robust = self.weighted_doubly_robust_sequential_policy_estimation(
            self.unshuffled_actions,
            self.unshuffled_rewards,
            self.unshuffled_terminals,
            self.unshuffled_logged_propensities,
            self.unshuffled_target_propensities,
            self.unshuffled_estimated_q_values,
            num_j_steps=Evaluator.NUM_J_STEPS_FOR_MAGIC_ESTIMATOR,
            whether_self_normalize_importance_weights=True,
        )
        self.value_magic_doubly_robust.append(magic_doubly_robust)

        logger.info(
            "Value Magic Doubly Robust P.E. : normalized {0:.3f} raw {1:.3f}".format(
                magic_doubly_robust.normalized, magic_doubly_robust.raw
            )
        )

        logger.info("CPE Evaluator Finished")

    def get_recent_td_loss(self):
        return Evaluator.calculate_recent_window_average(
            self.td_loss, Evaluator.RECENT_WINDOW_SIZE, num_entries=1
        )

    def get_recent_mc_loss(self):
        return Evaluator.calculate_recent_window_average(
            self.mc_loss, Evaluator.RECENT_WINDOW_SIZE, num_entries=1
        )

    def get_recent_reward_inverse_propensity_score(self):
        ips = Evaluator.calculate_recent_window_average(
            self.reward_inverse_propensity_score,
            Evaluator.RECENT_WINDOW_SIZE,
            num_entries=2,
        )
        return CPE_Estimate(normalized=ips[0], raw=ips[1])

    def get_recent_reward_direct_method(self):
        dm = Evaluator.calculate_recent_window_average(
            self.reward_direct_method, Evaluator.RECENT_WINDOW_SIZE, num_entries=2
        )
        return CPE_Estimate(normalized=dm[0], raw=dm[1])

    def get_recent_reward_doubly_robust(self):
        dr = Evaluator.calculate_recent_window_average(
            self.reward_doubly_robust, Evaluator.RECENT_WINDOW_SIZE, num_entries=2
        )
        return CPE_Estimate(normalized=dr[0], raw=dr[1])

    def get_recent_value_one_step_doubly_robust(self):
        dr = Evaluator.calculate_recent_window_average(
            self.value_doubly_robust, Evaluator.RECENT_WINDOW_SIZE, num_entries=2
        )
        return CPE_Estimate(normalized=dr[0], raw=dr[1])

    def get_recent_value_sequential_doubly_robust(self):
        dr = Evaluator.calculate_recent_window_average(
            self.value_sequential_doubly_robust, window_size=1, num_entries=2
        )
        return CPE_Estimate(normalized=dr[0], raw=dr[1])

    def get_recent_value_weighted_doubly_robust(self):
        dr = Evaluator.calculate_recent_window_average(
            self.value_weighted_doubly_robust, window_size=1, num_entries=2
        )
        return CPE_Estimate(normalized=dr[0], raw=dr[1])

    def get_recent_value_magic_doubly_robust(self):
        dr = Evaluator.calculate_recent_window_average(
            self.value_magic_doubly_robust, window_size=1, num_entries=2
        )
        return CPE_Estimate(normalized=dr[0], raw=dr[1])

    @staticmethod
    def calculate_recent_window_average(arr, window_size, num_entries):
        if len(arr) > 0:
            begin = max(0, len(arr) - window_size)
            return np.mean(np.array(arr[begin:]), axis=0)
        else:
            logger.error("Not enough samples for evaluation.")
            if num_entries == 1:
                return float("nan")
            else:
                return [float("nan")] * num_entries

    def doubly_robust_one_step_policy_estimation(
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
            CPE_Estimate(
                normalized=float(np.sum(ips) / total_reward), raw=float(np.mean(ips))
            ),
            CPE_Estimate(
                normalized=float(np.sum(direct_method_values) / total_reward),
                raw=float(np.mean(direct_method_values)),
            ),
            CPE_Estimate(
                normalized=float(np.sum(doubly_robust) / total_reward),
                raw=float(np.mean(doubly_robust)),
            ),
        )

    def doubly_robust_sequential_policy_estimation(
        self,
        logged_actions,
        logged_rewards,
        logged_terminals,
        logged_propensities,
        target_propensities,
        estimated_q_values,
    ):
        # For details, visit https://arxiv.org/pdf/1511.03722.pdf
        logged_terminals = logged_terminals.squeeze()
        logged_rewards = logged_rewards.squeeze()
        logged_propensities = logged_propensities.squeeze()

        num_examples = logged_actions.shape[0]

        estimated_state_values = np.sum(
            target_propensities * estimated_q_values, axis=1
        )
        estimated_q_values_for_logged_action = np.sum(
            estimated_q_values * logged_actions, axis=1
        )

        target_propensity_for_action = np.sum(
            target_propensities * logged_actions, axis=1
        )
        importance_weight = target_propensity_for_action / logged_propensities

        doubly_robusts = []
        episode_values = []

        i = 0
        last_episode_end = -1
        while i < num_examples:
            # calculate the doubly-robust Q-value for one episode
            if logged_terminals[i] or i == num_examples - 1:
                episode_end = i
                episode_value = 0.0
                doubly_robust = 0.0
                for j in range(episode_end, last_episode_end, -1):
                    doubly_robust = estimated_state_values[j] + importance_weight[j] * (
                        logged_rewards[j]
                        + self.gamma * doubly_robust
                        - estimated_q_values_for_logged_action[j]
                    )
                    episode_value *= self.gamma
                    episode_value += logged_rewards[j]
                if episode_value > 1e-6 or episode_value < -1e-6:
                    doubly_robusts.append(doubly_robust)
                    episode_values.append(episode_value)
                last_episode_end = episode_end
            i += 1

        assert len(episode_values) > 0, "Did not find any episode values"

        doubly_robusts = np.array(doubly_robusts)
        episode_values = np.array(episode_values)

        return CPE_Estimate(
            normalized=float(np.nanmean(doubly_robusts / episode_values)),
            raw=float(np.mean(doubly_robusts)),
        )

    def weighted_doubly_robust_sequential_policy_estimation(
        self,
        logged_actions,
        logged_rewards,
        logged_terminals,
        logged_propensities,
        target_propensities,
        estimated_q_values,
        num_j_steps,
        whether_self_normalize_importance_weights,
    ):
        # For details, visit https://arxiv.org/pdf/1604.00923.pdf Section 5, 7, 8
        (
            actions,
            rewards,
            logged_propensities,
            target_propensities,
            estimated_q_values,
        ) = Evaluator.transform_to_equal_length_trajectories(
            logged_terminals.squeeze(),
            logged_actions,
            logged_rewards.squeeze(),
            logged_propensities.squeeze(),
            target_propensities,
            estimated_q_values,
        )

        num_trajectories = actions.shape[0]
        trajectory_length = actions.shape[1]

        j_steps = [float("inf")]

        if num_j_steps > 1:
            j_steps.append(-1)
        if num_j_steps > 2:
            interval = trajectory_length // (num_j_steps - 1)
            j_steps.extend([i * interval for i in range(1, num_j_steps - 1)])

        target_propensity_for_logged_action = np.sum(
            np.multiply(target_propensities, actions), axis=2
        )
        estimated_q_values_for_logged_action = np.sum(
            np.multiply(estimated_q_values, actions), axis=2
        )
        estimated_state_values = np.sum(
            np.multiply(target_propensities, estimated_q_values), axis=2
        )

        importance_weights = target_propensity_for_logged_action / logged_propensities
        importance_weights = np.cumprod(importance_weights, axis=1)
        importance_weights = Evaluator.normalize_importance_weights(
            importance_weights, whether_self_normalize_importance_weights
        )

        importance_weights_one_earlier = (
            np.ones([num_trajectories, 1]) * 1.0 / num_trajectories
        )
        importance_weights_one_earlier = np.hstack(
            [importance_weights_one_earlier, importance_weights[:, :-1]]
        )

        discounts = np.logspace(
            start=0, stop=trajectory_length - 1, num=trajectory_length, base=self.gamma
        )

        j_step_return_trajectories = []
        for j_step in j_steps:
            j_step_return_trajectories.append(
                Evaluator.calculate_step_return(
                    rewards,
                    discounts,
                    importance_weights,
                    importance_weights_one_earlier,
                    estimated_state_values,
                    estimated_q_values_for_logged_action,
                    j_step,
                )
            )
        j_step_return_trajectories = np.array(j_step_return_trajectories)

        j_step_returns = np.sum(j_step_return_trajectories, axis=1)

        if len(j_step_returns) == 1:
            weighted_doubly_robust = j_step_returns[0]

        else:
            # break trajectories into several subsets to estimate confidence bounds
            infinite_step_returns = []
            num_subsets = int(
                min(num_trajectories / 2, Evaluator.NUM_SUBSETS_FOR_CB_ESTIMATES)
            )
            interval = num_trajectories / num_subsets
            for i in range(num_subsets):
                trajectory_subset = np.arange(
                    int(i * interval), int((i + 1) * interval)
                )
                importance_weights = (
                    target_propensity_for_logged_action[trajectory_subset]
                    / logged_propensities[trajectory_subset]
                )
                importance_weights = np.cumprod(importance_weights, axis=1)
                importance_weights = Evaluator.normalize_importance_weights(
                    importance_weights, whether_self_normalize_importance_weights
                )
                importance_weights_one_earlier = (
                    np.ones([len(trajectory_subset), 1]) * 1.0 / len(trajectory_subset)
                )
                importance_weights_one_earlier = np.hstack(
                    [importance_weights_one_earlier, importance_weights[:, :-1]]
                )
                infinite_step_return = np.sum(
                    Evaluator.calculate_step_return(
                        rewards[trajectory_subset],
                        discounts,
                        importance_weights,
                        importance_weights_one_earlier,
                        estimated_state_values[trajectory_subset],
                        estimated_q_values_for_logged_action[trajectory_subset],
                        float("inf"),
                    )
                )
                infinite_step_returns.append(infinite_step_return)

            low_bound, high_bound = Evaluator.confidence_bounds(
                infinite_step_returns, Evaluator.CONFIDENCE_INTERVAL
            )

            # decompose error into bias + variance
            j_step_bias = np.zeros([num_j_steps])
            where_lower = np.where(j_step_returns < low_bound)[0]
            j_step_bias[where_lower] = low_bound - j_step_returns[where_lower]
            where_higher = np.where(j_step_returns > high_bound)[0]
            j_step_bias[where_higher] = j_step_returns[where_higher] - high_bound

            covariance = np.cov(j_step_return_trajectories)

            error = covariance + j_step_bias.T * j_step_bias

            # minimize mse error
            def mse_loss(x, error):
                return np.dot(np.dot(x, error), x.T)

            constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

            x = np.zeros([len(j_steps)])
            res = sp.optimize.minimize(
                mse_loss,
                x,
                args=error,
                constraints=constraint,
                bounds=[(0, 1) for _ in range(x.shape[0])],
            )
            x = np.array(res.x)

            weighted_doubly_robust = np.dot(x, j_step_returns)

        episode_values = np.sum(np.multiply(rewards, discounts), axis=1)

        return CPE_Estimate(
            normalized=float(weighted_doubly_robust / np.nanmean(episode_values)),
            raw=float(weighted_doubly_robust),
        )

    @staticmethod
    def normalize_importance_weights(
        importance_weights, whether_self_normalize_importance_weights
    ):
        if whether_self_normalize_importance_weights:
            sum_importance_weights = np.sum(importance_weights, axis=0)
            where_zeros = np.where(sum_importance_weights == 0.0)[0]
            sum_importance_weights[where_zeros] = len(importance_weights)
            importance_weights[:, where_zeros] = 1.0
            importance_weights /= sum_importance_weights
            return importance_weights
        else:
            importance_weights /= importance_weights.shape[0]
            return importance_weights

    @staticmethod
    def calculate_step_return(
        rewards,
        discounts,
        importance_weights,
        importance_weights_one_earlier,
        estimated_state_values,
        estimated_q_values,
        j_step,
    ):
        trajectory_length = len(rewards[0])
        num_trajectories = len(rewards)
        j_step = int(min(j_step, trajectory_length - 1))

        weighted_discounts = np.multiply(discounts, importance_weights)
        weighted_discounts_one_earlier = np.multiply(
            discounts, importance_weights_one_earlier
        )

        importance_sampled_cumulative_reward = np.sum(
            np.multiply(weighted_discounts[:, : j_step + 1], rewards[:, : j_step + 1]),
            axis=1,
        )

        if j_step < trajectory_length - 1:
            direct_method_value = (
                weighted_discounts_one_earlier[:, j_step + 1]
                * estimated_state_values[:, j_step + 1]
            )
        else:
            direct_method_value = np.zeros([num_trajectories])

        control_variate = np.sum(
            np.multiply(
                weighted_discounts[:, : j_step + 1], estimated_q_values[:, : j_step + 1]
            )
            - np.multiply(
                weighted_discounts_one_earlier[:, : j_step + 1],
                estimated_state_values[:, : j_step + 1],
            ),
            axis=1,
        )

        j_step_return = (
            importance_sampled_cumulative_reward + direct_method_value - control_variate
        )

        return j_step_return

    @staticmethod
    def confidence_bounds(x, confidence):
        n = len(x)
        m, se = np.mean(x), sp.stats.sem(x)
        h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
        return m - h, m + h

    @staticmethod
    def transform_to_equal_length_trajectories(
        terminals,
        actions,
        rewards,
        logged_propensities,
        target_propensities,
        estimated_q_values,
    ):
        """
        Take into samples (action, rewards, propensities, etc.) and output lists
        of equal-length trajectories (episodes) accoriding to terminals.
        As the raw trajectories are of various lengths, the shorter ones are
        filled with zeros(ones) at the end.
        """
        num_actions = len(target_propensities[0])

        trajectories = []
        episode_start = 0
        episode_ends = np.nonzero(terminals)[0]
        if len(terminals) - 1 not in episode_ends:
            episode_ends = np.append(episode_ends, len(terminals) - 1)

        for episode_end in episode_ends:
            trajectories.append(np.arange(episode_start, episode_end + 1))
            episode_start = episode_end + 1

        action_trajectories = []
        reward_trajectories = []
        logged_propensity_trajectories = []
        target_propensity_trajectories = []
        Q_value_trajectories = []

        for trajectory in trajectories:
            action_trajectories.append(actions[trajectory])
            reward_trajectories.append(rewards[trajectory])
            logged_propensity_trajectories.append(logged_propensities[trajectory])
            target_propensity_trajectories.append(target_propensities[trajectory])
            Q_value_trajectories.append(estimated_q_values[trajectory])

        def to_equal_length(x, fill_value):
            x_equal_length = np.array(
                list(itertools.zip_longest(*x, fillvalue=fill_value))
            ).swapaxes(0, 1)
            return x_equal_length

        action_trajectories = to_equal_length(
            action_trajectories, np.zeros([num_actions])
        )
        reward_trajectories = to_equal_length(reward_trajectories, 0)
        logged_propensity_trajectories = to_equal_length(
            logged_propensity_trajectories, 1
        )
        target_propensity_trajectories = to_equal_length(
            target_propensity_trajectories, np.zeros([num_actions])
        )
        Q_value_trajectories = to_equal_length(
            Q_value_trajectories, np.zeros([num_actions])
        )

        return (
            action_trajectories,
            reward_trajectories,
            logged_propensity_trajectories,
            target_propensity_trajectories,
            Q_value_trajectories,
        )

    def log_to_tensorboard(self, writer: SummaryWriter, epoch: int) -> None:
        def none_to_zero(x: Optional[float]) -> float:
            if x is None or math.isnan(x):
                return 0.0
            return x

        for name, value in [
            ("Training/td_loss", self.get_recent_td_loss()),
            ("Training/mc_loss", self.get_recent_mc_loss()),
            (
                "Reward_CPE/Direct Method Reward",
                self.get_recent_reward_direct_method().normalized,
            ),
            (
                "Reward_CPE/IPS Reward",
                self.get_recent_reward_inverse_propensity_score().normalized,
            ),
            (
                "Reward_CPE/Doubly Robust Reward",
                self.get_recent_reward_doubly_robust().normalized,
            ),
            (
                "Value_CPE/MAGIC Estimator",
                self.get_recent_value_magic_doubly_robust().normalized,
            ),
            (
                "Value_CPE/Doubly Robust One Step",
                self.get_recent_value_one_step_doubly_robust().normalized,
            ),
            (
                "Value_CPE/Weighted Doubly Robust",
                self.get_recent_value_weighted_doubly_robust().normalized,
            ),
            (
                "Value_CPE/Sequential Doubly Robust",
                self.get_recent_value_sequential_doubly_robust().normalized,
            ),
        ]:
            writer.add_scalar(name, none_to_zero(value), epoch)

    @staticmethod
    def softmax(x, temperature):
        """Compute softmax values for each sets of scores in x."""
        x = x / temperature
        x -= np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    @staticmethod
    def huberLoss(label, output):
        if abs(label - output) > 1:
            return abs(label - output) - 0.5
        else:
            return 0.5 * (label - output) * (label - output)
