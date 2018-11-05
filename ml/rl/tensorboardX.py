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

from tensorboardX import SummaryWriter


class SummaryWriterContextMeta(type):
    def __getattr__(cls, func):
        if not cls._writer_stacks:

            def noop(*args, **kwargs):
                return

            return noop

        writer = cls._writer_stacks[-1]
        return getattr(writer, func)


class SummaryWriterContext(metaclass=SummaryWriterContextMeta):
    _writer_stacks = []

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
