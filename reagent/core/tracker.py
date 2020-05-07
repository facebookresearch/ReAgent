#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List

import torch


logger = logging.getLogger(__name__)


class Observer:
    """
    Base class for observers
    """

    def __init__(self, observing_keys: List[str]):
        super().__init__()
        assert isinstance(observing_keys, list)
        self.observing_keys = observing_keys

    def get_observing_keys(self) -> List[str]:
        return self.observing_keys

    def update(self, key: str, value):
        pass


class Aggregator:
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def __call__(self, key: str, values):
        assert key == self.key, f"Got {key}; expected {self.key}"
        self.aggregate(values)

    def aggregate(self, values):
        pass


def observable(cls=None, **kwargs):
    """
    Decorator to mark a class as producing observable values. The names of the
    observable values are the names of keyword arguments. The values of keyword
    arguments are the types of the value. The type is currently not used for
    anything.
    """
    assert kwargs
    observable_value_types = kwargs

    def wrap(cls):
        assert not hasattr(cls, "add_observer")
        assert not hasattr(cls, "notify_observers")

        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            assert not hasattr(self, "_observable_value_types")
            assert not hasattr(self, "_observers")
            self._observable_value_types = observable_value_types
            self._observers = {v: [] for v in observable_value_types}

        cls.__init__ = new_init

        def add_observer(self, observer: Observer) -> None:
            observing_keys = observer.get_observing_keys()
            unknown_keys = [
                k for k in observing_keys if k not in self._observable_value_types
            ]
            if unknown_keys:
                logger.warning(f"{unknown_keys} cannot be observed in {type(self)}")
            for k in observing_keys:
                if k in self._observers and observer not in self._observers[k]:
                    self._observers[k].append(observer)
            return self

        cls.add_observer = add_observer

        def add_observers(self, observers: List[Observer]) -> None:
            for observer in observers:
                self.add_observer(observer)
            return self

        cls.add_observers = add_observers

        def notify_observers(self, **kwargs):
            for key, value in kwargs.items():
                if value is None:
                    # Allow optional reporting
                    continue

                assert key in self._observers, f"Unknown key: {key}"

                # TODO: Create a generic framework for type conversion
                if self._observable_value_types[key] == torch.Tensor:
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value)
                    if len(value.shape) == 0:
                        value = value.reshape(1)
                    value = value.detach()

                for observer in self._observers[key]:
                    observer.update(key, value)

        cls.notify_observers = notify_observers

        return cls

    if cls is None:
        return wrap

    return wrap(cls)
