#!/usr/bin/env python3

import itertools
import logging
from typing import List, Optional

import torch
from reagent.core import aggregators as agg
from reagent.core.rl_training_output import RLTrainingOutput
from reagent.core.union import TrainingReport__Union
from reagent.reporting.oss_training_reports import OssDQNTrainingReport
from reagent.reporting.reporter_base import ReporterBase


logger = logging.getLogger(__name__)


class DiscreteDQNReporter(ReporterBase):
    def __init__(
        self,
        actions: List[str],
        report_interval: int = 100,
        target_action_distribution: Optional[List[float]] = None,
        recent_window_size: int = 100,
    ):
        aggregators = itertools.chain(
            [
                ("CPE Results", agg.AppendAggregator("cpe_details")),
                ("TD Loss", agg.MeanAggregator("td_loss", interval=report_interval)),
                (
                    "Reward Loss",
                    agg.MeanAggregator("reward_loss", interval=report_interval),
                ),
                (
                    "Model Action Values",
                    agg.FunctionsByActionAggregator(
                        "model_values",
                        actions,
                        {"mean": torch.mean, "std": torch.std},
                        interval=report_interval,
                    ),
                ),
                (
                    "Logged Actions",
                    agg.ActionCountAggregator(
                        "logged_actions", actions, interval=report_interval
                    ),
                ),
                (
                    "model_action",
                    agg.ActionCountAggregator(
                        "model_action_idxs", actions, interval=report_interval
                    ),
                ),
                (
                    "Recent Logged Rewards",
                    agg.RecentValuesAggregator(
                        "logged_rewards", interval=report_interval
                    ),
                ),
            ],
            [
                (
                    f"{key}_tb",
                    agg.TensorBoardActionCountAggregator(
                        key, title, actions, interval=report_interval
                    ),
                )
                for key, title in [
                    ("logged_actions", "logged"),
                    ("model_action_idxs", "model"),
                ]
            ],
            [
                (
                    f"{key}_tb",
                    agg.TensorBoardHistogramAndMeanAggregator(
                        key, log_key, interval=report_interval
                    ),
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
                        key, category, title, actions, interval=report_interval
                    ),
                )
                for key, category, title in [
                    ("model_propensities", "propensities", "model"),
                    ("model_rewards", "reward", "model"),
                    ("model_values", "value", "model"),
                ]
            ],
        )
        super().__init__(aggregators)
        self.target_action_distribution = target_action_distribution
        self.recent_window_size = recent_window_size

    def publish(self) -> RLTrainingOutput:
        return RLTrainingOutput(
            training_report=TrainingReport__Union(oss_dqn_report=OssDQNTrainingReport())
        )
