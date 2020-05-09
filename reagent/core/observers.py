#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict, Iterable, List, Optional

from reagent.core.tracker import Aggregator, Observer


logger = logging.getLogger(__name__)


class CompositeObserver(Observer):
    """
    A composite observer which takes care of dispatching values to child observers
    """

    def __init__(self, observers: Iterable[Observer]):
        self.observers: Dict[str, List[Observer]] = {}
        for observer in observers:
            observing_keys = observer.get_observing_keys()
            for key in observing_keys:
                self.observers.setdefault(key, []).append(observer)
        super().__init__(list(self.observers))

    def update(self, key: str, value):
        for observer in self.observers[key]:
            observer.update(key, value)


class EpochEndObserver(Observer):
    """
    Call the callback function with epoch # when the epoch ends
    """

    def __init__(self, callback, key: str = "epoch_end"):
        super().__init__(observing_keys=[key])
        self.callback = callback

    def update(self, key: str, value):
        self.callback(value)


class ValueListObserver(Observer):
    """
    Simple observer that collect values into a list
    """

    def __init__(self, observing_key: str):
        super().__init__(observing_keys=[observing_key])
        self.observing_key = observing_key
        self.values: List[Any] = []

    def update(self, key: str, value):
        self.values.append(value)

    def reset(self):
        self.values = []


class IntervalAggregatingObserver(Observer):
    def __init__(self, interval: Optional[int], aggregator: Aggregator):
        self.key = aggregator.key
        super().__init__(observing_keys=["epoch_end", self.key])
        self.iteration = 0
        self.interval = interval
        self.intermediate_values: List[Any] = []
        self.aggregator = aggregator

    def update(self, key: str, value):
        if key == "epoch_end":
            self.flush()
            return

        self.intermediate_values.append(value)
        self.iteration += 1
        # pyre-fixme[6]: Expected `int` for 1st param but got `Optional[int]`.
        if self.interval and self.iteration % self.interval == 0:
            logger.info(
                f"Interval Agg. Update: {self.key}; iteration {self.iteration}; "
                f"aggregator: {self.aggregator.__class__.__name__}"
            )
            self.aggregator(self.key, self.intermediate_values)
            self.intermediate_values = []

    def flush(self):
        # We need to reset iteration here to avoid aggregating on the same data multiple
        # times
        logger.info(
            f"Interval Agg. Flushing: {self.key}; iteration: {self.iteration}; "
            f"aggregator: {self.aggregator.__class__.__name__}; points: {len(self.intermediate_values)}"
        )
        self.iteration = 0
        if self.intermediate_values:
            self.aggregator(self.key, self.intermediate_values)
        self.intermediate_values = []
