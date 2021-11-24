#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import logging
from typing import Dict, Optional, Type

from reagent.core.dataclasses import dataclass
from reagent.core.fb_checker import IS_FB_ENVIRONMENT


logger = logging.getLogger(__name__)


class RegistryMeta(abc.ABCMeta):
    def __init__(cls, name, bases, attrs):
        if not hasattr(cls, "REGISTRY"):
            # Put REGISTRY on cls. This only happens once on the base class
            logger.info("Adding REGISTRY to type {}".format(name))
            cls.REGISTRY: Dict[str, Type] = {}
            cls.REGISTRY_NAME = name
            cls.REGISTRY_FROZEN = False

        assert not cls.REGISTRY_FROZEN, (
            f"{cls.REGISTRY_NAME} has been used to fill a union. "
            "Please rearrange your import orders"
        )

        if not cls.__abstractmethods__ and name != cls.REGISTRY_NAME:
            # Only register fully-defined classes
            logger.info(f"Registering {name} to {cls.REGISTRY_NAME}")
            if hasattr(cls, "__registry_name__"):
                registry_name = cls.__registry_name__
                logger.info(f"Using {registry_name} instead of {name}")
                name = registry_name
            assert name not in cls.REGISTRY, f"{name} in REGISTRY {cls.REGISTRY}"
            cls.REGISTRY[name] = cls
        else:
            logger.info(
                f"Not Registering {name} to {cls.REGISTRY_NAME}. Abstract "
                f"method {list(cls.__abstractmethods__)} are not implemented."
            )
        return super().__init__(name, bases, attrs)

    def fill_union(cls):
        def wrapper(union):
            cls.REGISTRY_FROZEN = True

            def make_union_instance(inst, instance_class=None):
                inst_class = instance_class or type(inst)
                key = getattr(inst_class, "__registry_name__", inst_class.__name__)
                return union(**{key: inst})

            union.make_union_instance = make_union_instance

            if not IS_FB_ENVIRONMENT:
                # OSS TaggedUnion
                union.__annotations__ = {
                    name: Optional[t] for name, t in cls.REGISTRY.items()
                }
                for name in cls.REGISTRY:
                    setattr(union, name, None)
                return dataclass(frozen=True)(union)
            else:
                # FBL TaggedUnion
                union.__annotations__ = {name: t for name, t in cls.REGISTRY.items()}
                return union

        return wrapper


def wrap_oss_with_dataclass(union):
    if not IS_FB_ENVIRONMENT:
        # OSS TaggedUnion
        return dataclass(frozen=True)(union)
    else:
        # FBL TaggedUnion
        return union
