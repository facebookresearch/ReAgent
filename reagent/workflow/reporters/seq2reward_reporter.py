#!/usr/bin/env python3

import itertools
import logging
from typing import List

import torch
from reagent.core import aggregators as agg
from reagent.core.observers import IntervalAggregatingObserver
from reagent.workflow.reporters.reporter_base import ReporterBase
from reagent.workflow.training_reports import Seq2RewardTrainingReport


logger = logging.getLogger(__name__)


class Seq2RewardReporter(ReporterBase):
    def __init__(self, action_names: List[str], report_interval: int = 100):
        self.action_names = action_names
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
                    ("mse_loss_per_batch", agg.MeanAggregator("mse_loss")),
                    (
                        "step_entropy_loss_per_batch",
                        agg.MeanAggregator("step_entropy_loss"),
                    ),
                    (
                        "q_values_per_batch",
                        agg.FunctionsByActionAggregator(
                            "q_values", self.action_names, {"mean": torch.mean}
                        ),
                    ),
                    ("eval_mse_loss_per_batch", agg.MeanAggregator("eval_mse_loss")),
                    (
                        "eval_step_entropy_loss_per_batch",
                        agg.MeanAggregator("eval_step_entropy_loss"),
                    ),
                    (
                        "eval_q_values_per_batch",
                        agg.FunctionsByActionAggregator(
                            "eval_q_values", self.action_names, {"mean": torch.mean}
                        ),
                    ),
                    (
                        "eval_action_distribution_per_batch",
                        agg.FunctionsByActionAggregator(
                            "eval_action_distribution",
                            self.action_names,
                            {"mean": torch.mean},
                        ),
                    ),
                ],
                [
                    (
                        f"{key}_tb",
                        agg.TensorBoardHistogramAndMeanAggregator(key, log_key),
                    )
                    for key, log_key in [
                        ("mse_loss", "mse_loss"),
                        ("step_entropy_loss", "step_entropy_loss"),
                        ("eval_mse_loss", "eval_mse_loss"),
                        ("eval_step_entropy_loss", "eval_step_entropy_loss"),
                    ]
                ],
                [
                    (
                        f"{key}_tb",
                        agg.TensorBoardActionHistogramAndMeanAggregator(
                            key, category, title, self.action_names
                        ),
                    )
                    for key, category, title in [
                        ("q_values", "q_values", "training"),
                        ("eval_q_values", "q_values", "eval"),
                        ("eval_action_distribution", "action_distribution", "eval"),
                    ]
                ],
            )
        }

    # TODO: write this for OSS
    def generate_training_report(self) -> Seq2RewardTrainingReport:
        return Seq2RewardTrainingReport()


class Seq2RewardCompressReporter(Seq2RewardReporter):
    @property
    def aggregating_observers(self):
        return {
            name: IntervalAggregatingObserver(self.report_interval, aggregator)
            for name, aggregator in itertools.chain(
                [
                    ("mse_loss_per_batch", agg.MeanAggregator("mse_loss")),
                    ("accuracy_per_batch", agg.MeanAggregator("accuracy")),
                    ("eval_mse_loss_per_batch", agg.MeanAggregator("eval_mse_loss")),
                    ("eval_accuracy_per_batch", agg.MeanAggregator("eval_accuracy")),
                    (
                        "eval_q_values_per_batch",
                        agg.FunctionsByActionAggregator(
                            "eval_q_values", self.action_names, {"mean": torch.mean}
                        ),
                    ),
                    (
                        "eval_action_distribution_per_batch",
                        agg.FunctionsByActionAggregator(
                            "eval_action_distribution",
                            self.action_names,
                            {"mean": torch.mean},
                        ),
                    ),
                ],
                [
                    (
                        f"{key}_tb",
                        agg.TensorBoardHistogramAndMeanAggregator(key, log_key),
                    )
                    for key, log_key in [
                        ("mse_loss", "compress_mse_loss"),
                        ("accuracy", "compress_accuracy"),
                        ("eval_mse_loss", "compress_eval_mse_loss"),
                        ("eval_accuracy", "compress_eval_accuracy"),
                    ]
                ],
                [
                    (
                        f"{key}_tb",
                        agg.TensorBoardActionHistogramAndMeanAggregator(
                            key, category, title, self.action_names
                        ),
                    )
                    for key, category, title in [
                        ("eval_q_values", "q_values", "compress_eval"),
                        (
                            "eval_action_distribution",
                            "action_distribution",
                            "compress_eval",
                        ),
                    ]
                ],
            )
        }
