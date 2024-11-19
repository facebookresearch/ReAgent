#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import abc
import os
import unittest

from reagent.core.configuration import make_config_class, resolve_defaults
from reagent.core.dataclasses import dataclass, field
from reagent.core.registry_meta import RegistryMeta
from reagent.core.tagged_union import TaggedUnion


class A:
    @resolve_defaults
    def __init__(
        self,
        a: int = 1,
        b: int = field(default_factory=lambda: 2),  # noqa
    ) -> None:
        self.a = a
        self.b = b

    def __call__(self) -> int:
        return self.a * self.b


@make_config_class(A.__init__)
class AParam:
    pass


class FooRegistry(metaclass=RegistryMeta):
    @abc.abstractmethod
    def foo(self) -> int:
        pass


@dataclass
class Foo(FooRegistry):
    a_param: AParam = field(default_factory=AParam)

    def foo(self):
        a = A(**self.a_param.asdict())
        return a()


@dataclass
class Bar(FooRegistry):
    def foo(self) -> int:
        return 10


@FooRegistry.fill_union()
class FooUnion(TaggedUnion):
    pass


@dataclass
class Config:
    union: FooUnion = field(
        # pyre-fixme[28]: Unexpected keyword argument `Foo`.
        default_factory=lambda: FooUnion(Foo=Foo())
    )


class TestConfigParsing(unittest.TestCase):
    def test_parse_foo_default(self) -> None:
        raw_config = {}
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 2)

    def test_parse_foo(self) -> None:
        raw_config = {"union": {"Foo": {"a_param": {"a": 6}}}}
        # pyre-fixme[6]: For 1st param expected `FooUnion` but got `Dict[str,
        #  Dict[str, Dict[str, int]]]`.
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 12)

    def test_parse_bar(self) -> None:
        raw_config = {"union": {"Bar": {}}}
        # pyre-fixme[6]: For 1st param expected `FooUnion` but got `Dict[str,
        #  Dict[typing.Any, typing.Any]]`.
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 10)

    def test_frozen_registry(self) -> None:
        with self.assertRaises(RuntimeError):

            @dataclass
            class Baz(FooRegistry):
                def foo(self):
                    return 20

        self.assertListEqual(sorted(FooRegistry.REGISTRY.keys()), ["Bar", "Foo"])

    def test_frozen_registry_skip(self) -> None:
        _environ = dict(os.environ)
        os.environ.update({"SKIP_FROZEN_REGISTRY_CHECK": "1"})
        try:

            @dataclass
            class Baz(FooRegistry):
                def foo(self):
                    return 20

        finally:
            os.environ.clear()
            os.environ.update(_environ)

        self.assertListEqual(sorted(FooRegistry.REGISTRY.keys()), ["Bar", "Foo"])
