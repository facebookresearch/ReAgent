#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
Context library to allow dropping tensorboardX anywhere in the codebase.
If there is no SummaryWriter in the context, function calls will be no-op.

Usage:

    writer = SummaryWriter()

    with summary_writer_context(writer):
        some_func()


    def some_func():
        SummaryWriterContext.add_scalar("foo", tensor)

"""

import contextlib
import logging

from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class SummaryWriterContextMeta(type):
    def __getattr__(cls, func):
        if not cls._writer_stacks:

            def noop(*args, **kwargs):
                return

            return noop

        writer = cls._writer_stacks[-1]

        def call(*args, **kwargs):
            if "global_step" not in kwargs:
                kwargs["global_step"] = cls._global_step
            try:
                return getattr(writer, func)(*args, **kwargs)
            except Exception as e:
                if hasattr(writer, "exceptions_to_ignore") and isinstance(
                    e, writer.exceptions_to_ignore
                ):
                    logger.warning("Ignoring exception: {}".format(e))
                    if hasattr(writer, "exception_logging_func"):
                        writer.exception_logging_func(e)
                    return
                raise

        return call


class SummaryWriterContext(metaclass=SummaryWriterContextMeta):
    _writer_stacks = []
    _global_step = 0
    _custom_scalars = {}

    @classmethod
    def _reset_globals(cls):
        cls._global_step = 0
        cls._custom_scalars = {}

    @classmethod
    def increase_global_step(cls):
        cls._global_step += 1

    @classmethod
    def add_custom_scalars(cls, writer):
        """
        Call this once you are satisfied setting up custom scalar
        """
        writer.add_custom_scalars(cls._custom_scalars)

    @classmethod
    def add_custom_scalars_multilinechart(cls, tags, category=None, title=None):
        assert category and title, "category & title must be set"
        if category not in cls._custom_scalars:
            cls._custom_scalars[category] = {}
        assert (
            title not in cls._custom_scalars[category]
        ), "Title ({}) is already in category ({})".format(title, category)
        cls._custom_scalars[category][title] = ["Multiline", tags]

    @classmethod
    def push(cls, writer):
        assert isinstance(
            writer, SummaryWriter
        ), "writer is not a SummaryWriter: {}".format(writer)
        cls._writer_stacks.append(writer)

    @classmethod
    def pop(cls):
        return cls._writer_stacks.pop()


@contextlib.contextmanager
def summary_writer_context(writer):
    if writer is not None:
        SummaryWriterContext.push(writer)
    try:
        yield
    finally:
        if writer is not None:
            SummaryWriterContext.pop()
