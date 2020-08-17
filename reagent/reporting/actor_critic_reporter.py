#!/usr/bin/env python3

import itertools
import logging

from reagent.core import aggregators as agg
from reagent.core.types import RLTrainingOutput, TrainingReport__Union
from reagent.reporting.oss_training_reports import OssActorCriticTrainingReport
from reagent.reporting.reporter_base import ReporterBase


logger = logging.getLogger(__name__)


class ActorCriticReporter(ReporterBase):
    def __init__(self, report_interval: int = 100):
        aggregators = itertools.chain(
            [
                ("cpe_results", agg.AppendAggregator("cpe_details")),
                ("td_loss", agg.MeanAggregator("td_loss", interval=report_interval)),
                (
                    "reward_loss",
                    agg.MeanAggregator("reward_loss", interval=report_interval),
                ),
                (
                    "recent_rewards",
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

    # TODO: T71636196 write this for OSS
    def publish(self) -> RLTrainingOutput:
        report = OssActorCriticTrainingReport()
        return RLTrainingOutput(
            training_report=TrainingReport__Union(oss_actor_critic_report=report)
        )
