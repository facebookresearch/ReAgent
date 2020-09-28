#!/usr/bin/env python3

import abc
import logging
from typing import Dict

from reagent.core.observers import (
    CompositeObserver,
    EpochEndObserver,
    IntervalAggregatingObserver,
    ValueListObserver,
)
from reagent.workflow.result_registries import TrainingReport


logger = logging.getLogger(__name__)


class ReporterBase(CompositeObserver):
    def __init__(
        self,
        value_list_observers: Dict[str, ValueListObserver],
        aggregating_observers: Dict[str, IntervalAggregatingObserver],
    ):
        epoch_end_observer = EpochEndObserver(self._epoch_end_callback)
        self.last_epoch_end_num_batches: int = 0
        self.num_data_points_per_epoch = None
        super().__init__(
            list(value_list_observers.values())
            # pyre-fixme[58]: `+` is not supported for operand types
            #  `List[ValueListObserver]` and `List[IntervalAggregatingObserver]`.
            # pyre-fixme[58]: `+` is not supported for operand types
            #  `List[ValueListObserver]` and `List[IntervalAggregatingObserver]`.
            + list(aggregating_observers.values())
            # pyre-fixme[58]: `+` is not supported for operand types
            #  `List[ValueListObserver]` and `List[EpochEndObserver]`.
            # pyre-fixme[58]: `+` is not supported for operand types
            #  `List[ValueListObserver]` and `List[EpochEndObserver]`.
            + [epoch_end_observer]
        )

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
    @abc.abstractmethod
    def generate_training_report(self) -> TrainingReport:
        pass
