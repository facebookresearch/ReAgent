#!/usr/bin/env python3

import itertools
import logging
from collections import OrderedDict

from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver, ValueListObserver
from reagent.reporting.reporter_printer import ReporterPrinter
from reagent.workflow.reporters.reporter_base import ReporterBase
from reagent.workflow.training_reports import ParametricDQNTrainingReport


logger = logging.getLogger(__name__)


class ParametricDQNReporter(ReporterBase):
    def __init__(self, report_interval: int = 100):
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
                            "recent_rewards",
                            agg.RecentValuesAggregator("logged_rewards"),
                        ),
                        (
                            "model_values_on_logged_actions",
                            agg.MeanAggregator("model_values_on_logged_actions"),
                        ),
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
                )
            },
        }
        super().__init__(self.value_list_observers, self.aggregating_observers)

    # TODO: write this for OSS
    def generate_training_report(
        self, reporter_printer: ReporterPrinter
    ) -> ParametricDQNTrainingReport:
        return ParametricDQNTrainingReport()
