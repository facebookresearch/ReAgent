#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging

from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver
from reagent.reporting.reporter_base import ReporterBase

# pyre-fixme[21]: Could not find module `reagent.workflow.training_reports`.
from reagent.workflow.training_reports import ActorCriticTrainingReport


logger = logging.getLogger(__name__)


class PolicyGradientReporter(ReporterBase):
    """Collects training metrics for policy-gradient trainers (PPO/Reinforce)."""

    def __init__(self, report_interval: int = 1):
        self.report_interval = report_interval
        super().__init__(self.value_list_observers, self.aggregating_observers)

    @property
    def value_list_observers(self):
        return {}

    @property
    def aggregating_observers(self):
        return {
            name: IntervalAggregatingObserver(self.report_interval, aggregator)
            for name, aggregator in [
                (
                    f"{key}_tb",
                    agg.TensorBoardHistogramAndMeanAggregator(key, log_key),
                )
                for key, log_key in [
                    ("ppo_loss", "loss/ppo_loss"),
                    ("value_net_loss", "loss/value_net_loss"),
                ]
            ]
        }

    # pyre-fixme[15]: `generate_training_report` overrides method defined in
    #  `ReporterBase` inconsistently.
    # pyre-fixme[11]: Annotation `ActorCriticTrainingReport` is not defined as a type.
    def generate_training_report(self) -> ActorCriticTrainingReport:
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        return ActorCriticTrainingReport()
