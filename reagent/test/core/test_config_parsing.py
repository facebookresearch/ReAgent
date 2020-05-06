#!/usr/bin/env python3

import abc
import unittest

from reagent.core.configuration import make_config_class, resolve_defaults
from reagent.core.dataclasses import dataclass, field
from reagent.core.registry_meta import RegistryMeta
from reagent.core.tagged_union import TaggedUnion


class A:
    @resolve_defaults
    def __init__(
        self, a: int = 1, b: int = field(default_factory=lambda: 2)  # noqa
    ) -> None:
        self.a = a
        self.b = b

    def __call__(self):
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
    def foo(self):
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
    def test_parse_foo_default(self):
        raw_config = {}
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 2)

    def test_parse_foo(self):
        raw_config = {"union": {"Foo": {"a_param": {"a": 6}}}}
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 12)

    def test_parse_bar(self):
        raw_config = {"union": {"Bar": {}}}
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 10)
