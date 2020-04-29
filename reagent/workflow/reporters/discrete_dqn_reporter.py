#!/usr/bin/env python3

import logging
from collections import OrderedDict
from typing import List, Optional

import torch
from reagent.core import aggregators as agg
from reagent.core.observers import (
    CompositeObserver,
    EpochEndObserver,
    IntervalAggregatingObserver,
    ValueListObserver,
)
from reagent.workflow.training_reports import DQNTrainingReport


logger = logging.getLogger(__name__)


# TODO(T64634239): Create OSS version of DiscreteDQNReporter
class DiscreteDQNReporter(CompositeObserver):
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
            for name, aggregator in [
                ("td_loss", agg.MeanAggregator("td_loss")),
                ("reward_loss", agg.MeanAggregator("reward_loss")),
                (
                    "model_values",
                    agg.FunctionsByActionAggregator(
                        "model_values", actions, {"mean": torch.mean, "std": torch.std}
                    ),
                ),
                ("logged_action", agg.ActionCountAggregator("logged_actions", actions)),
                (
                    "model_action",
                    agg.ActionCountAggregator("model_action_idxs", actions),
                ),
                ("recent_rewards", agg.RecentValuesAggregator("logged_rewards")),
            ]
            + [
                (f"{key}_tb", agg.TensorBoardActionCountAggregator(key, title, actions))
                for key, title in [
                    ("logged_actions", "logged"),
                    ("model_action_idxs", "model"),
                ]
            ]
            + [
                (f"{key}_tb", agg.TensorBoardHistogramAndMeanAggregator(key, log_key))
                for key, log_key in [
                    ("td_loss", "td_loss"),
                    ("reward_loss", "reward_loss"),
                    ("logged_propensities", "propensities/logged"),
                    ("logged_rewards", "reward/logged"),
                ]
            ]
            + [
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
            ]
        )
        self.last_epoch_end_num_batches = 0
        self.num_data_points_per_epoch = None
        epoch_end_observer = EpochEndObserver(self._epoch_end_callback)
        super().__init__(
            list(self.value_list_observers.values())
            + list(self.aggregating_observers.values())
            + [epoch_end_observer]
        )
        self.target_action_distribution = target_action_distribution
        self.recent_window_size = recent_window_size

    def _epoch_end_callback(self, epoch: int):
        logger.info(f"Epoch {epoch} ended")

        for observer in self.aggregating_observers.values():
            observer.flush()

        num_batches = len(self.td_loss.values) - self.last_epoch_end_num_batches
        self.last_epoch_end_num_batches = len(self.td_loss.values)
        if self.num_data_points_per_epoch is None:
            self.num_data_points_per_epoch = num_batches
        else:
            assert self.num_data_points_per_epoch == num_batches
        logger.info(f"Epoch {epoch} contains {num_batches} aggregated data points")

    def __getattr__(self, key: str):
        if key in self.value_list_observers:
            return self.value_list_observers[key]
        return self.aggregating_observers[key].aggregator

    # TODO: write this for OSS
    def generate_training_report(self) -> DQNTrainingReport:
        return DQNTrainingReport()

    # TODO: Delete this method once we don't use EvaluationPageHandler in
    # discrete DQN
    def report(self, evaluation_details):
        evaluation_details.log_to_tensorboard()
