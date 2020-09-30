#!/usr/bin/env python3

import abc
import logging
from typing import Dict, Optional

import torch
from pytorch_lightning.utilities import rank_zero_only
from reagent.core.observers import (
    CompositeObserver,
    EpochEndObserver,
    IntervalAggregatingObserver,
    ValueListObserver,
)
from reagent.core.tracker import ObservableMixin
from reagent.core.utils import lazy_property
from reagent.workflow.result_registries import TrainingReport


logger = logging.getLogger(__name__)


class ReporterBase(CompositeObserver):
    def __init__(
        self,
        value_list_observers: Dict[str, ValueListObserver],
        aggregating_observers: Dict[str, IntervalAggregatingObserver],
    ):
        epoch_end_observer = EpochEndObserver(self.flush)
        self.last_epoch_end_num_batches: int = 0
        self.num_data_points_per_epoch: Optional[int] = None
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
        self._reporter_observable = _ReporterObservable(self)

    @rank_zero_only
    def log(self, **kwargs) -> None:
        self._reporter_observable.notify_observers(**kwargs)

    @rank_zero_only
    def flush(self, epoch: int):
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


class _ReporterObservable(ObservableMixin):
    def __init__(self, reporter) -> None:
        self._reporter = reporter
        super().__init__()
        self.add_observer(reporter)

    @lazy_property
    def _observable_value_types(self):
        return {k: torch.Tensor for k in self._reporter.get_observing_keys()}
