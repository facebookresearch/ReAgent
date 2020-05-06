#!/usr/bin/env python3


import dataclasses
import logging
import os

# Redirection to make import simpler
from dataclasses import field  # noqa
from typing import TYPE_CHECKING, Optional

import pydantic


try:
    import fblearner.flow.api  # noqa

    """
    Inside FBLearner, we don't use pydantic for option parsing. Some types don't have
    validator. This necessary to avoid pydantic complaining about validators.
    """
    USE_VANILLA_DATACLASS = True

except ImportError:

    USE_VANILLA_DATACLASS = False


ARBITRARY_TYPES_ALLOWED = False


try:
    # Allowing override, e.g., in unit test
    USE_VANILLA_DATACLASS = bool(int(os.environ["USE_VANILLA_DATACLASS"]))
except KeyError:
    pass

try:
    ARBITRARY_TYPES_ALLOWED = bool(int(os.environ["ARBITRARY_TYPES_ALLOWED"]))
except KeyError:
    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info(f"USE_VANILLA_DATACLASS: {USE_VANILLA_DATACLASS}")
logger.info(f"ARBITRARY_TYPES_ALLOWED: {ARBITRARY_TYPES_ALLOWED}")


if TYPE_CHECKING:
    """
    HACK: FB's mypy doesn't like the wrapper below and throws a bunch of type errors
    """
    from dataclasses import dataclass

else:

    def dataclass(
        _cls: Optional[pydantic.typing.AnyType] = None, *, config=None, **kwargs
    ):
        def wrap(cls):
            # We don't want to look at parent class
            if "__post_init__" in cls.__dict__:
                raise TypeError(
                    f"{cls} has __post_init__. "
                    "Please use __post_init_post_parse__ instead."
                )

            if USE_VANILLA_DATACLASS:
                try:
                    post_init_post_parse = cls.__dict__["__post_init_post_parse__"]
                    logger.info(
                        f"Setting {cls.__name__}.__post_init__ to its "
                        "__post_init_post_parse__"
                    )
                    cls.__post_init__ = post_init_post_parse
                except KeyError:
                    pass

                return dataclasses.dataclass(**kwargs)(cls)
            else:
                if ARBITRARY_TYPES_ALLOWED:

                    class Config:
                        arbitrary_types_allowed = ARBITRARY_TYPES_ALLOWED

                    assert config not in kwargs
                    kwargs["config"] = Config

                return pydantic.dataclasses.dataclass(cls, **kwargs)

        if _cls is None:
            return wrap

        return wrap(_cls)
