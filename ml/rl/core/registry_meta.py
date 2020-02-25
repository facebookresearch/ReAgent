#!/usr/bin/env python3

import abc
import logging
from typing import Dict, Type


try:
    from fblearner.flow.api import types
except ImportError:
    pass


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

    def canonical_flow_type(cls):
        struct = types.struct.input_schema_to_struct(cls.__init__)
        logger.info(f"Configuration type for {cls}: {struct}")
        return struct

    def registry_union_type(cls):
        logger.info(f"Creating union type for {cls}")

        class Union:
            @classmethod
            def canonical_flow_type(cls):
                return types.UNION(**cls.__annotations__)

        Union.__annotations__ = {
            name: t.canonical_flow_type() for name, t in cls.REGISTRY.items()
        }

        return Union

    def create(cls, name: str, config):
        assert name in cls.REGISTRY, "{} is not registered; use one of {}".format(
            name, cls.REGISTRY.keys()
        )
        logger.info("Creating instance of {} from config: {}".format(name, config))
        is_dict = isinstance(config, dict)
        if is_dict or hasattr(config, "items"):
            if not is_dict:
                config = dict(config.items())
            return cls.REGISTRY[name](**config)
        return cls.REGISTRY[name](config)

    def create_from_union(cls, union):
        return cls.create(union.selected_field, union.value)
