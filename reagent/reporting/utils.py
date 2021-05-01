#!/usr/bin/env python3

import logging
import math

import numpy as np
import scipy  # @manual=third-party//scipy:scipy-py
from pandas import DataFrame
from reagent.evaluation.cpe import CpeDetails

logger = logging.getLogger(__name__)


def _compute_action_kl_divergence(eval_dist, train_dist):
    """
    Compute D_KL(train_dist || eval_dist).
    """
    if eval_dist is None:
        return None
    pk = []
    qk = []
    for k in eval_dist:
        pk.append(train_dist[k])
        # Put a small value to avoid inf
        qk.append(max(eval_dist[k], 1e-12))
    return scipy.stats.entropy(pk, qk)


def _compute_q_value_kl_divergence(q_value_means, cpe_results):
    """
    Compute sum of KL divergence between train && eval q-values of each action.
    This assumes q-values have gaussian distribution.
    """
    if cpe_results.q_value_means is None:
        return None
    try:
        kl_divergence = 0.0
        for k in cpe_results.q_value_means:
            eval_mean = cpe_results.q_value_means[k]
            eval_std = max(cpe_results.q_value_stds[k], 1e-12)
            train_mean = q_value_means[k]
            train_std = max(q_value_means["{}_std".format(k)], 1e-12)
            logger.info(
                "Stats for action {}; train mean {} std {}; eval mean {} std {}".format(
                    k, train_mean, train_std, eval_mean, eval_std
                )
            )
            kl_divergence += (
                math.log(eval_std)
                - math.log(train_std)
                + (train_std ** 2 + (train_mean - eval_mean) ** 2) / (2 * eval_std ** 2)
                - 0.5
            )
        return kl_divergence
    except ValueError:
        return None


def convert_mc_loss_estimate_to_schema_type(estimate):
    if estimate is None:
        return np.nan
    return estimate


def generate_metrics_table(evaluation_details: CpeDetails):
    """Returns table of metrics and CPE scores."""
    data = []
    for metric, estimate_set in evaluation_details.metric_estimates.items():
        if (
            estimate_set.direct_method is None
            or estimate_set.inverse_propensity is None
            or estimate_set.doubly_robust is None
            or estimate_set.sequential_doubly_robust is None
            or estimate_set.weighted_doubly_robust is None
            or estimate_set.magic is None
        ):
            raise AssertionError("Missing necessary CPE values")

        table_row = [
            metric,
            estimate_set.direct_method.normalized,
            estimate_set.inverse_propensity.normalized,
            estimate_set.doubly_robust.normalized,
            estimate_set.sequential_doubly_robust.normalized,
            estimate_set.weighted_doubly_robust.normalized,
            estimate_set.magic.normalized,
        ]
        data.append(table_row)
    df = DataFrame(
        data,
        columns=[
            "Metric",
            "Direct Method",
            "Inverse Propensity",
            "Doubly Robust",
            "Sequential Doubly Robust",
            "Weighted Doubly Robust",
            "Magic",
        ],
    )
    return df
