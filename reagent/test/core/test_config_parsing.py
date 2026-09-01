#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import abc
import unittest
import unittest.mock as mock

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


def _new_foo_registry() -> tuple[type, type]:
    class FooRegistry(metaclass=RegistryMeta):
        @abc.abstractmethod
        def foo(self) -> int:
            pass

    @dataclass
    class Foo(FooRegistry):
        a_param: AParam = field(default_factory=AParam)

        def foo(self) -> int:
            a = A(**self.a_param.asdict())
            return a()

    @dataclass
    class Bar(FooRegistry):
        def foo(self) -> int:
            return 10

    @FooRegistry.fill_union()
    # pyrefly: ignore [invalid-inheritance]
    class FooUnion(TaggedUnion):
        pass

    @dataclass
    class Config:
        union: FooUnion = field(default_factory=lambda: FooUnion(Foo=Foo()))

    return FooRegistry, Config


class TestConfigParsing(unittest.TestCase):
    def test_parse_foo_default(self) -> None:
        _, Config = _new_foo_registry()
        raw_config = {}
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 2)

    def test_parse_foo(self) -> None:
        _, Config = _new_foo_registry()
        raw_config = {"union": {"Foo": {"a_param": {"a": 6}}}}
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 12)

    def test_parse_bar(self) -> None:
        _, Config = _new_foo_registry()
        raw_config = {"union": {"Bar": {}}}
        config = Config(**raw_config)
        self.assertEqual(config.union.value.foo(), 10)

    @mock.patch("reagent.core.registry_meta.skip_frozen_registry_check")
    def test_frozen_registry(
        self, mock_skip_frozen_registry_check: mock.MagicMock
    ) -> None:
        FooRegistry, _ = _new_foo_registry()
        mock_skip_frozen_registry_check.return_value = False

        with self.assertRaises(RuntimeError) as context:

            @dataclass
            class Baz(FooRegistry):
                def foo(self) -> int:
                    return 20

        self.assertIn(
            "FooRegistry has been used to fill a union and is now frozen, so Baz can't be added to the registry.",
            str(context.exception),
        )
        self.assertListEqual(sorted(FooRegistry.REGISTRY.keys()), ["Bar", "Foo"])

    @mock.patch("reagent.core.registry_meta.skip_frozen_registry_check")
    def test_frozen_registry_skip(
        self, mock_skip_frozen_registry_check: mock.MagicMock
    ) -> None:
        FooRegistry, _ = _new_foo_registry()
        mock_skip_frozen_registry_check.return_value = True

        @dataclass
        class Baz(FooRegistry):
            def foo(self) -> int:
                return 20

        self.assertListEqual(sorted(FooRegistry.REGISTRY.keys()), ["Bar", "Baz", "Foo"])
