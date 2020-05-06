#!/usr/bin/env python3

import abc
import logging
from typing import Iterable

from reagent.core.observers import CompositeObserver
from reagent.core.tracker import Observer
from reagent.workflow.result_registries import TrainingReport


logger = logging.getLogger(__name__)


class ReporterBase(CompositeObserver):
    def __init__(self, observers: Iterable[Observer]):
        super().__init__(observers)
        self.num_data_points_per_epoch = None
        self.last_epoch_end_num_batches: int = 0

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
