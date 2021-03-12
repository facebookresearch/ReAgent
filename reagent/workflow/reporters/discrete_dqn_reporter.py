#!/usr/bin/env python3

import itertools
import logging
from typing import List, Optional

import torch
from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver, ValueListObserver
from reagent.core.report_utils import (
    calculate_recent_window_average,
    get_mean_of_recent_values,
)
from reagent.reporting.training_reports import DQNTrainingReport
from reagent.reporting.utils import (
    _compute_action_kl_divergence,
    _compute_q_value_kl_divergence,
    convert_mc_loss_estimate_to_schema_type,
)
from reagent.reporting.utils import (
    generate_metrics_table,
)
from reagent.workflow.reporters.reporter_base import (
    ReporterBase,
    FlexibleDataPointsPerEpochMixin,
)


logger = logging.getLogger(__name__)


class DiscreteDQNReporter(FlexibleDataPointsPerEpochMixin, ReporterBase):
    def __init__(
        self,
        actions: List[str],
        report_interval: int = 100,
        target_action_distribution: Optional[List[float]] = None,
        recent_window_size: int = 100,
    ):
        self.value_list_observers = {}
        self.aggregating_observers = {
            **{
                "cpe_results": IntervalAggregatingObserver(
                    1, agg.ListAggregator("cpe_details")
                ),
            },
            **{
                name: IntervalAggregatingObserver(report_interval, aggregator)
                for name, aggregator in itertools.chain(
                    [
                        ("td_loss", agg.MeanAggregator("td_loss")),
                        ("reward_loss", agg.MeanAggregator("reward_loss")),
                        (
                            "model_values",
                            agg.FunctionsByActionAggregator(
                                "model_values",
                                actions,
                                {"mean": torch.mean, "std": torch.std},
                            ),
                        ),
                        (
                            "logged_action",
                            agg.ActionCountAggregator("logged_actions", actions),
                        ),
                        (
                            "model_action",
                            agg.ActionCountAggregator("model_action_idxs", actions),
                        ),
                        (
                            "recent_rewards",
                            agg.RecentValuesAggregator("logged_rewards"),
                        ),
                    ],
                    [
                        (
                            f"{key}_tb",
                            agg.TensorBoardActionCountAggregator(key, title, actions),
                        )
                        for key, title in [
                            ("logged_actions", "logged"),
                            ("model_action_idxs", "model"),
                        ]
                    ],
                    [
                        (
                            f"{key}_tb",
                            agg.TensorBoardHistogramAndMeanAggregator(key, log_key),
                        )
                        for key, log_key in [
                            ("td_loss", "td_loss"),
                            ("eval_td_loss", "eval_td_loss"),
                            ("reward_loss", "reward_loss"),
                            ("logged_propensities", "propensities/logged"),
                            ("logged_rewards", "reward/logged"),
                        ]
                    ],
                    [
                        (
                            f"{key}_tb",
                            agg.TensorBoardActionHistogramAndMeanAggregator(
                                key, category, title, actions
                            ),
                        )
                        for key, category, title in [
                            ("model_propensities", "propensities", "model"),
                            ("model_rewards", "reward", "model"),
                            ("model_values", "value", "model"),
                        ]
                    ],
                )
            },
        }
        super().__init__(self.value_list_observers, self.aggregating_observers)
        self.target_action_distribution = target_action_distribution
        self.recent_window_size = recent_window_size

    def generate_training_report(self) -> DQNTrainingReport:
        last_cpe_results = self._get_last_cpe_results()
        cpe_metrics_table = generate_metrics_table(last_cpe_results)
        zeroed_reward_estimates = (
            last_cpe_results.reward_estimates.fill_empty_with_zero()
        )

        all_model_action_distributions = self.model_action.get_distributions()
        model_action_distribution = get_mean_of_recent_values(
            all_model_action_distributions
        )
        target_distribution_error = self._get_target_distribution_error(
            model_action_distribution
        )
        q_value_means = self._get_average_q_values_by_action()
        q_value_kl_divergence = _compute_q_value_kl_divergence(
            q_value_means, last_cpe_results
        )
        logged_action_distribution = self.logged_action.get_cumulative_distributions()

        return DQNTrainingReport(
            td_loss=calculate_recent_window_average(
                self.td_loss.values, self.recent_window_size, num_entries=1
            ),
            mc_loss=convert_mc_loss_estimate_to_schema_type(last_cpe_results.mc_loss),
            reward_ips=zeroed_reward_estimates.inverse_propensity,
            reward_dm=zeroed_reward_estimates.direct_method,
            reward_dr=zeroed_reward_estimates.doubly_robust,
            value_sequential_dr=zeroed_reward_estimates.sequential_doubly_robust,
            value_weighted_dr=zeroed_reward_estimates.weighted_doubly_robust,
            value_magic_dr=zeroed_reward_estimates.magic,
            cpe_metrics_table=cpe_metrics_table,
            logged_action_distribution=logged_action_distribution,
            model_action_distribution=model_action_distribution,
            model_logged_dist_kl_divergence=_compute_action_kl_divergence(
                logged_action_distribution, model_action_distribution
            ),
            target_distribution_error=target_distribution_error,
            q_value_means=q_value_means,
            eval_q_value_means=last_cpe_results.q_value_means,
            eval_q_value_stds=last_cpe_results.q_value_stds,
            eval_action_distribution=last_cpe_results.action_distribution,
            q_value_kl_divergence=q_value_kl_divergence,
            action_dist_kl_divergence=_compute_action_kl_divergence(
                last_cpe_results.action_distribution, model_action_distribution
            ),
        )
