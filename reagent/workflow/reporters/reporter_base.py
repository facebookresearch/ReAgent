#!/usr/bin/env python3

import abc
import logging
from typing import Dict

import torch
from pytorch_lightning.utilities import rank_zero_only
from reagent.core.observers import (
    CompositeObserver,
    EpochEndObserver,
    IntervalAggregatingObserver,
    ValueListObserver,
)
from reagent.core.result_registries import TrainingReport
from reagent.core.tracker import ObservableMixin
from reagent.core.utils import lazy_property
from reagent.reporter import Reporter
from reagent.types import ReportData, ReportEntry


logger = logging.getLogger(__name__)


class ReporterBase(CompositeObserver, Reporter):
    def __init__(
        self,
        value_list_observers: Dict[str, ValueListObserver],
        aggregating_observers: Dict[str, IntervalAggregatingObserver],
    ):
        epoch_end_observer = EpochEndObserver(self.flush)
        self._value_list_observers = value_list_observers
        self._aggregating_observers = aggregating_observers
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

        for observer in self._aggregating_observers.values():
            observer.flush()

    @rank_zero_only
    def generate_report(self) -> ReportData:
        rd = ReportData(
            entries=[
                observer.generate_report_entry()
                for observer in self._aggregating_observers.values()
            ]
            + [
                observer.generate_report_entry()
                for observer in self._value_list_observers.values()
            ]
        )
        return rd

    def __getattr__(self, key: str):
        val = self._value_list_observers.get(key, None)
        if val is not None:
            return val
        val = self._aggregating_observers.get(key, None)
        if val is not None:
            return val.aggregator
        raise AttributeError


class _ReporterObservable(ObservableMixin):
    def __init__(self, reporter) -> None:
        self._reporter = reporter
        super().__init__()
        self.add_observer(reporter)

    @lazy_property
    def _observable_value_types(self):
        return {k: torch.Tensor for k in self._reporter.get_observing_keys()}


class DataPointsPerEpochMixin(ReporterBase):
    """
    The reporter should have td_loss as value list to use this
    """

    @rank_zero_only
    def flush(self, epoch: int):
        super().flush(epoch)
        try:
            last_epoch_end_num_batches = self.last_epoch_end_num_batches
            num_data_points_per_epoch = self.num_data_points_per_epoch
        except AttributeError:
            last_epoch_end_num_batches = 0
            num_data_points_per_epoch = None

        num_batches = len(self.td_loss.values) - last_epoch_end_num_batches
        setattr(self, "last_epoch_end_num_batches", len(self.td_loss.values))
        if num_data_points_per_epoch is None:
            setattr(self, "num_data_points_per_epoch", num_batches)
        else:
            assert num_data_points_per_epoch == num_batches
        logger.info(f"Epoch {epoch} contains {num_batches} aggregated data points")


class FlexibleDataPointsPerEpochMixin(ReporterBase):
    """
    Similar to DataPointsPerEpochMixin, but does not enforce the same number of batches
    across epochs to allow for variable length trajectories
    """

    @rank_zero_only
    def flush(self, epoch: int):
        super().flush(epoch)
        try:
            last_epoch_end_num_batches = self.last_epoch_end_num_batches
            num_data_points_per_epoch = self.num_data_points_per_epoch
        except AttributeError:
            last_epoch_end_num_batches = 0
            num_data_points_per_epoch = None

        num_batches = len(self.td_loss.values) - last_epoch_end_num_batches
        setattr(self, "last_epoch_end_num_batches", len(self.td_loss.values))
        setattr(self, "num_data_points_per_epoch", num_batches)
        logger.info(f"Epoch {epoch} contains {num_batches} aggregated data points")
