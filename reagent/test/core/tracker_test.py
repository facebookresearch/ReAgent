#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

from reagent.core.observers import ValueListObserver
from reagent.core.tracker import observable


class TestObservable(unittest.TestCase):
    def test_observable(self):
        @observable(td_loss=float, str_val=str)
        class DummyClass:
            def __init__(self, a, b, c=10):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def do_something(self, i):
                self.notify_observers(td_loss=i, str_val="not_used")

        instance = DummyClass(1, 2)
        self.assertIsInstance(instance, DummyClass)
        self.assertEqual(instance.a, 1)
        self.assertEqual(instance.b, 2)
        self.assertEqual(instance.c, 10)

        observers = [ValueListObserver("td_loss") for _i in range(3)]
        instance.add_observers(observers)
        # Adding twice should not result in double update
        instance.add_observer(observers[0])

        for i in range(10):
            instance.do_something(float(i))

        for observer in observers:
            self.assertEqual(observer.values, [float(i) for i in range(10)])

    def test_no_observable_values(self):
        try:

            @observable()
            class NoObservableValues:
                pass

        except AssertionError:
            pass
