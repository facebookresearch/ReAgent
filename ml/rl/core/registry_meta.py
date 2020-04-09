#!/usr/bin/env python3

import abc
import logging
from typing import Dict, Optional, Type

from ml.rl.core.dataclasses import dataclass
from ml.rl.core.tagged_union import TaggedUnion


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RegistryMeta(abc.ABCMeta):
    def __init__(cls, name, bases, attrs):
        if not hasattr(cls, "REGISTRY"):
            # Put REGISTRY on cls. This only happens once on the base class
            logger.info("Adding REGISTRY to type {}".format(name))
            cls.REGISTRY: Dict[str, Type] = {}
            cls.REGISTRY_NAME = name

        if not cls.__abstractmethods__:
            # Only register fully-defined classes
            logger.info("Registering {} to {}".format(name, cls.REGISTRY_NAME))
            cls.REGISTRY[name] = cls
        else:
            logger.info(
                f"Not Registering {name} to {cls.REGISTRY_NAME}. Abstract "
                f"method {list(cls.__abstractmethods__)} are not implemented."
            )
        return super().__init__(name, bases, attrs)

    def fill_union(cls):
        def wrapper(union):
            if issubclass(union, TaggedUnion):
                # OSS TaggedUnion
                union.__annotations__ = {
                    name: Optional[t] for name, t in cls.REGISTRY.items()
                }
                for name in cls.REGISTRY:
                    setattr(union, name, None)
                return dataclass(frozen=True)(union)

            # FBL TaggedUnion
            union.__annotations__ = {name: t for name, t in cls.REGISTRY.items()}
            return union

        return wrapper
