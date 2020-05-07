#!/usr/bin/env python3

import itertools
import logging
from collections import OrderedDict
from typing import List, Optional

import torch
from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver, ValueListObserver
from reagent.workflow.reporters.reporter_base import ReporterBase
from reagent.workflow.training_reports import DQNTrainingReport


logger = logging.getLogger(__name__)


class DiscreteDQNReporter(ReporterBase):
    def __init__(
        self,
        actions: List[str],
        report_interval: int = 100,
        target_action_distribution: Optional[List[float]] = None,
        recent_window_size: int = 100,
    ):
        self.value_list_observers = {"cpe_results": ValueListObserver("cpe_details")}
        self.aggregating_observers = OrderedDict(
            (name, IntervalAggregatingObserver(report_interval, aggregator))
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
                    ("recent_rewards", agg.RecentValuesAggregator("logged_rewards")),
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
        )
        super().__init__(self.value_list_observers, self.aggregating_observers)
        self.target_action_distribution = target_action_distribution
        self.recent_window_size = recent_window_size

    # TODO: write this for OSS
    def generate_training_report(self) -> DQNTrainingReport:
        cpe_results = self.value_list_observers["cpe_results"].values  # noqa
        return DQNTrainingReport()
