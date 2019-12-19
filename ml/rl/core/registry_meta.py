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

    def create(cls, name: str, config):
        assert name in cls.REGISTRY, "{} is not registered; use one of {}".format(
            name, cls.REGISTRY.keys()
        )
        logger.info("Creating instance of {} from config: {}".format(name, config))
        return cls.REGISTRY[name](config)

    def create_from_union(cls, union):
        return cls.create(union.selected_field, union.value)
