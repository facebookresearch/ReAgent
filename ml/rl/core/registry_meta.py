#!/usr/bin/env python3

import abc
import logging
from typing import Dict, Type


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
            union.__annotations__ = {name: t for name, t in cls.REGISTRY.items()}
            return union

        return wrapper
