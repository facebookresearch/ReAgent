#!/usr/bin/env python3


# Redirection to make import simpler
from dataclasses import field  # noqa
from typing import TYPE_CHECKING, Optional

import pydantic


class Config:
    arbitrary_types_allowed = True


if TYPE_CHECKING:
    """
    HACK: FB's mypy doesn't like the wrapper below and throws a bunch of type errors
    """
    from dataclasses import dataclass

else:

    def dataclass(
        _cls: Optional[pydantic.typing.AnyType] = None, *, config=None, **kwargs
    ):
        """
        Inside FB, we don't use pydantic for option parsing. Some types don't have
        validator. This necessary to avoid pydantic complaining about validators.
        """

        if config is None:
            config = Config

        def wrap(cls):
            return pydantic.dataclasses.dataclass(cls, config=config, **kwargs)

        if _cls is None:
            return wrap

        return wrap(_cls)
