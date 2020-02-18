#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import List, Type


class Observer:
    """
    Base class for observers
    """

    def update(self, key: str, value):
        pass


class ValueListObserver(Observer):
    """
    Simple observer that collect values into a list
    """

    def __init__(self, observing_values: List[str]):
        super().__init__()
        self.values = {v: [] for v in observing_values}

    def update(self, key: str, value):
        self.values[key].append(value)


def observable(cls: Type = None, **kwargs) -> Type:
    """
    Decorator to mark a class as producing observable values. The names of the
    observable values are the names of keyword arguments. The values of keyword
    arguments are the types of the value. The type is currently not used for
    anything.
    """
    assert kwargs
    observable_values = kwargs

    def wrap(cls):
        assert not hasattr(cls, "add_observer")
        assert not hasattr(cls, "notify_observers")

        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            assert not hasattr(self, "_observable_values")
            assert not hasattr(self, "_observers")
            self._observable_values = observable_values
            self._observers = {v: [] for v in observable_values}

        cls.__init__ = new_init

        def add_observer(self, observer, observing_values: List[str]):
            unknown_values = [v for v in observing_values if v not in self._observers]
            assert not unknown_values, f"{unknown_values} cannot be observed"
            assert isinstance(observer, Observer)
            for v in observing_values:
                if observer not in self._observers[v]:
                    self._observers[v].append(observer)

        cls.add_observer = add_observer

        def notify_observers(self, **kwargs):
            for key, value in kwargs.items():
                assert key in self._observers, f"Unknown key: {key}"
                for observer in self._observers[key]:
                    observer.update(key, value)

        cls.notify_observers = notify_observers

        return cls

    if cls is None:
        return wrap

    return wrap(cls)
