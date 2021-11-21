#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools
import logging

from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver
from reagent.reporting.reporter_base import ReporterBase
from reagent.workflow.training_reports import WorldModelTrainingReport


logger = logging.getLogger(__name__)


class WorldModelReporter(ReporterBase):
    def __init__(self, report_interval: int = 100):
        self.report_interval = report_interval
        super().__init__(self.value_list_observers, self.aggregating_observers)

    @property
    def value_list_observers(self):
        return {}

    @property
    def aggregating_observers(self):
        return {
            name: IntervalAggregatingObserver(self.report_interval, aggregator)
            for name, aggregator in itertools.chain(
                [
                    ("loss", agg.MeanAggregator("loss")),
                    ("gmm", agg.MeanAggregator("gmm")),
                    ("bce", agg.MeanAggregator("bce")),
                    ("mse", agg.MeanAggregator("mse")),
                    ("eval_loss", agg.MeanAggregator("eval_loss")),
                    ("eval_gmm", agg.MeanAggregator("eval_gmm")),
                    ("eval_bce", agg.MeanAggregator("eval_bce")),
                    ("eval_mse", agg.MeanAggregator("eval_mse")),
                    ("test_loss", agg.MeanAggregator("test_loss")),
                    ("test_gmm", agg.MeanAggregator("test_gmm")),
                    ("test_bce", agg.MeanAggregator("test_bce")),
                    ("test_mse", agg.MeanAggregator("test_mse")),
                ],
                [
                    (
                        f"{key}_tb",
                        agg.TensorBoardHistogramAndMeanAggregator(key, log_key),
                    )
                    for key, log_key in [
                        ("loss", "loss"),
                        ("gmm", "gmm"),
                        ("bce", "bce"),
                        ("mse", "mse"),
                        ("eval_loss", "eval_loss"),
                        ("eval_gmm", "eval_gmm"),
                        ("eval_bce", "eval_bce"),
                        ("eval_mse", "eval_mse"),
                        ("test_loss", "test_loss"),
                        ("test_gmm", "test_gmm"),
                        ("test_bce", "test_bce"),
                        ("test_mse", "test_mse"),
                    ]
                ],
            )
        }

    # TODO: write this for OSS
    def generate_training_report(self) -> WorldModelTrainingReport:
        return WorldModelTrainingReport()
