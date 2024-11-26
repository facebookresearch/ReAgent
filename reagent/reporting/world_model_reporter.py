#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging

from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver
from reagent.reporting.reporter_base import ReporterBase

# pyre-fixme[21]: Could not find module `reagent.workflow.training_reports`.
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
        agg_obs = {}
        for name in [
            "loss",
            "gmm",
            "bce",
            "mse",
            "eval_loss",
            "eval_gmm",
            "eval_bce",
            "eval_mse",
            "test_loss",
            "test_gmm",
            "test_bce",
            "test_mse",
        ]:
            # Mean Aggegators - average losses over every report_interval minibatches
            mean_agg = agg.MeanAggregator(name)
            agg_obs[name] = IntervalAggregatingObserver(self.report_interval, mean_agg)
            # Tensorboard aggregators
            tb_obs_name = f"{name}_tb"
            tb_agg = agg.TensorBoardHistogramAndMeanAggregator(name, name)
            agg_obs[tb_obs_name] = IntervalAggregatingObserver(
                self.report_interval, tb_agg
            )
            # Epoch Aggregators - average losses per epoch
            ep_obs_name = f"{name}_epoch"
            ep_mean_agg = agg.MeanAggregator(name)
            agg_obs[ep_obs_name] = IntervalAggregatingObserver(
                999999999999999,  # a huge report interval to prevent from aggregating before epoch ends
                ep_mean_agg,
            )
        return agg_obs

    # TODO: write this for OSS
    # pyre-fixme[15]: `generate_training_report` overrides method defined in
    #  `ReporterBase` inconsistently.
    # pyre-fixme[11]: Annotation `WorldModelTrainingReport` is not defined as a type.
    def generate_training_report(self) -> WorldModelTrainingReport:
        # pyre-fixme[16]: Module `reagent` has no attribute `workflow`.
        return WorldModelTrainingReport()
