#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from collections import Counter
from typing import Dict, List, Optional

import torch
from ml.rl.evaluation.cpe import CpeDetails, CpeEstimateSet
from ml.rl.evaluation.doubly_robust_estimator import DoublyRobustEstimator
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.evaluation.sequential_doubly_robust_estimator import (
    SequentialDoublyRobustEstimator,
)
from ml.rl.evaluation.weighted_sequential_doubly_robust_estimator import (
    WeightedSequentialDoublyRobustEstimator,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_tensor(x, dtype=None):
    """
    Input:
        - x: list or a sequence
        - dtype: target data type of the elements in tensor [optional]
                 It will be infered automatically if not provided.
    Output:
        Tensor given a list or a sequence.
        If the input is None, it returns None
        If the input is a tensor it returns the tensor.
        If type is provides the output Tensor will have that type
    """
    if x is None:
        return None

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if dtype is not None:
        x = x.type(dtype)

    return x


def get_metrics_to_score(metric_reward_values: Optional[Dict[str, float]]) -> List[str]:
    if metric_reward_values is None:
        return []
    return sorted([*metric_reward_values.keys()])


class Evaluator(object):
    NUM_J_STEPS_FOR_MAGIC_ESTIMATOR = 25

    def __init__(self, action_names, gamma, model, metrics_to_score=None) -> None:
        self.action_names = action_names
        self.metrics_to_score = metrics_to_score

        self.gamma = gamma
        self.model = model

        self.doubly_robust_estimator = DoublyRobustEstimator()
        self.sequential_doubly_robust_estimator = SequentialDoublyRobustEstimator(gamma)
        self.weighted_sequential_doubly_robust_estimator = WeightedSequentialDoublyRobustEstimator(
            gamma
        )

    def evaluate_post_training(self, edp: EvaluationDataPage) -> CpeDetails:
        cpe_details = CpeDetails()

        cpe_details.reward_estimates = self.score_cpe("Reward", edp)

        if (
            self.metrics_to_score is not None
            and edp.logged_metrics is not None
            and self.action_names is not None
        ):
            for i, metric in enumerate(self.metrics_to_score):
                logger.info(
                    "--------- Running CPE on metric: {} ---------".format(metric)
                )

                metric_reward_edp = edp.set_metric_as_reward(i, len(self.action_names))

                cpe_details.metric_estimates[metric] = self.score_cpe(
                    metric, metric_reward_edp
                )

        if self.action_names is not None:
            if edp.optimal_q_values is not None:
                value_means = edp.optimal_q_values.mean(dim=0)
                cpe_details.q_value_means = {
                    action: float(value_means[i])
                    for i, action in enumerate(self.action_names)
                }
                value_stds = edp.optimal_q_values.std(dim=0)
                cpe_details.q_value_stds = {
                    action: float(value_stds[i])
                    for i, action in enumerate(self.action_names)
                }
            if edp.eval_action_idxs is not None:
                cpe_details.action_distribution = {
                    action: float((edp.eval_action_idxs == i).sum())
                    / edp.eval_action_idxs.shape[0]
                    for i, action in enumerate(self.action_names)
                }

        # Compute MC Loss on Aggregate Reward
        cpe_details.mc_loss = float(
            torch.mean(torch.abs(edp.logged_values - edp.model_values))
        )

        return cpe_details

    def score_cpe(self, metric_name, edp: EvaluationDataPage):
        direct_method, inverse_propensity, doubly_robust = self.doubly_robust_estimator.estimate(
            edp
        )
        sequential_doubly_robust = self.sequential_doubly_robust_estimator.estimate(edp)
        weighted_doubly_robust = self.weighted_sequential_doubly_robust_estimator.estimate(
            edp, num_j_steps=1, whether_self_normalize_importance_weights=True
        )
        magic = self.weighted_sequential_doubly_robust_estimator.estimate(
            edp,
            num_j_steps=Evaluator.NUM_J_STEPS_FOR_MAGIC_ESTIMATOR,
            whether_self_normalize_importance_weights=True,
        )
        return CpeEstimateSet(
            direct_method=direct_method,
            inverse_propensity=inverse_propensity,
            doubly_robust=doubly_robust,
            sequential_doubly_robust=sequential_doubly_robust,
            weighted_doubly_robust=weighted_doubly_robust,
            magic=magic,
        )

    def _get_batch_logged_actions(self, arr):
        action_counter = Counter()
        for actions in arr:
            # torch.max() returns the element and the index.
            # The latter is the argmax equivalent
            _, argmax = torch.max(actions, dim=1)

            # Counter object does not work well with Tensors, hence casting back to numpy
            action_counter.update(Counter(argmax.numpy()))

        total_actions = 1.0 * sum(action_counter.values())

        return (
            {
                action_name: (action_counter[i] / total_actions)
                for i, action_name in enumerate(self.action_names)
            },
            {
                action_name: action_counter[i]
                for i, action_name in enumerate(self.action_names)
            },
        )

    def get_target_distribution_error(
        self, actions, target_distribution, actual_distribution
    ):
        """Calculate MSE between actual and target action distribution."""
        if not target_distribution:
            return None
        error = 0
        for i, action in enumerate(actions):
            error += (target_distribution[i] - actual_distribution[action]) ** 2
        return error / len(actions)

    @staticmethod
    def huberLoss(label, output):
        if abs(label - output) > 1:
            return abs(label - output) - 0.5
        else:
            return 0.5 * (label - output) * (label - output)
