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


logger = logging.getLogger(__name__)


class ReporterBase(CompositeObserver):
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

    def __getattr__(self, key: str):
        val = self._value_list_observers.get(key, None)
        if val is not None:
            return val
        val = self._aggregating_observers.get(key, None)
        if val is not None:
            return val.aggregator
        raise AttributeError

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
