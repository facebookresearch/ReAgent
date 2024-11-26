#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import itertools
import logging

from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver, ValueListObserver
from reagent.reporting.reporter_base import ReporterBase

# pyre-fixme[21]: Could not find module `reagent.workflow.training_reports`.
from reagent.workflow.training_reports import SlateQTrainingReport


logger = logging.getLogger(__name__)


class SlateQReporter(ReporterBase):
    def __init__(self, report_interval: int = 100):
        self.report_interval = report_interval
        super().__init__(self.value_list_observers, self.aggregating_observers)

    @property
    def value_list_observers(self):
        return {"cpe_results": ValueListObserver("cpe_details")}

    @property
    def aggregating_observers(self):
        return {
            name: IntervalAggregatingObserver(self.report_interval, aggregator)
            for name, aggregator in itertools.chain(
                [
                    ("td_loss", agg.MeanAggregator("td_loss")),
                    ("recent_rewards", agg.RecentValuesAggregator("logged_rewards")),
                    (
                        "logged_action_q_value",
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
                        ("logged_rewards", "reward/logged"),
                    ]
                ],
            )
        }

    # pyre-fixme[15]: `generate_training_report` overrides method defined in
    #  `ReporterBase` inconsistently.
    # pyre-fixme[11]: Annotation `SlateQTrainingReport` is not defined as a type.
    def generate_training_report(self) -> SlateQTrainingReport:
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        return SlateQTrainingReport()
