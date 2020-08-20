#!/usr/bin/env python3

import itertools
import logging
from typing import List, Optional

from reagent.core import aggregators as agg
from reagent.core.rl_training_output import RLTrainingOutput
from reagent.core.union import TrainingReport__Union
from reagent.reporting.oss_training_reports import OssParametricDQNTrainingReport
from reagent.reporting.reporter_base import ReporterBase


logger = logging.getLogger(__name__)


class ParametricDQNReporter(ReporterBase):
    def __init__(
        self,
        report_interval: int = 100,
        target_action_distribution: Optional[List[float]] = None,
        recent_window_size: int = 100,
    ):
        aggregators = itertools.chain(
            [
                ("cpe_results", agg.AppendAggregator("cpe_results")),
                ("td_loss", agg.MeanAggregator("td_loss", interval=report_interval)),
                (
                    "reward_loss",
                    agg.MeanAggregator("reward_loss", interval=report_interval),
                ),
                (
                    "logged_rewards",
                    agg.RecentValuesAggregator(
                        "logged_rewards", interval=report_interval
                    ),
                ),
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
        )
        super().__init__(aggregators)
        self.target_action_distribution = target_action_distribution
        self.recent_window_size = recent_window_size

    # TODO: T71636218 write this for OSS
    def publish(self) -> RLTrainingOutput:
        cpe_results = self.cpe_results.values
        report = OssParametricDQNTrainingReport()
        return RLTrainingOutput(
            training_report=TrainingReport__Union(oss_parametric_dqn_report=report)
        )
