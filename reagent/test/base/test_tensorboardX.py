#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import call, MagicMock

import torch
from reagent.core.tensorboardX import summary_writer_context, SummaryWriterContext
from reagent.test.base.horizon_test_base import HorizonTestBase
from torch.utils.tensorboard import SummaryWriter


class TestSummaryWriterContext(HorizonTestBase):
    def test_noop(self) -> None:
        self.assertIsNone(SummaryWriterContext.add_scalar("test", torch.ones(1)))

    def test_with_none(self) -> None:
        with summary_writer_context(None):
            self.assertIsNone(SummaryWriterContext.add_scalar("test", torch.ones(1)))

    def test_writing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock()
            with summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))
            writer.add_scalar.assert_called_once_with(
                "test", torch.ones(1), global_step=0
            )

    def test_writing_stack(self) -> None:
        with TemporaryDirectory() as tmp_dir1, TemporaryDirectory() as tmp_dir2:
            writer1 = SummaryWriter(tmp_dir1)
            writer1.add_scalar = MagicMock()
            writer2 = SummaryWriter(tmp_dir2)
            writer2.add_scalar = MagicMock()
            with summary_writer_context(writer1):
                with summary_writer_context(writer2):
                    SummaryWriterContext.add_scalar("test2", torch.ones(1))
                SummaryWriterContext.add_scalar("test1", torch.zeros(1))
            writer1.add_scalar.assert_called_once_with(
                "test1", torch.zeros(1), global_step=0
            )
            writer2.add_scalar.assert_called_once_with(
                "test2", torch.ones(1), global_step=0
            )

    def test_swallowing_exception(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock(side_effect=NotImplementedError("test"))
            # pyre-fixme[16]: `SummaryWriter` has no attribute `exceptions_to_ignore`.
            writer.exceptions_to_ignore = (NotImplementedError, KeyError)
            with summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))

    def test_not_swallowing_exception(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock(side_effect=NotImplementedError("test"))
            with self.assertRaisesRegex(
                NotImplementedError, "test"
            ), summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))

    def test_swallowing_histogram_value_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            with summary_writer_context(writer):
                SummaryWriterContext.add_histogram("bad_histogram", torch.ones(100, 1))

    def test_global_step(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock()
            with summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))
                SummaryWriterContext.increase_global_step()
                SummaryWriterContext.add_scalar("test", torch.zeros(1))
            writer.add_scalar.assert_has_calls(
                [
                    call("test", torch.ones(1), global_step=0),
                    call("test", torch.zeros(1), global_step=1),
                ]
            )
            self.assertEqual(2, len(writer.add_scalar.mock_calls))

    def test_add_custom_scalars(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_custom_scalars = MagicMock()
            with summary_writer_context(writer):
                SummaryWriterContext.add_custom_scalars_multilinechart(
                    ["a", "b"], category="cat", title="title"
                )
                with self.assertRaisesRegex(
                    AssertionError, "Title \\(title\\) is already in category \\(cat\\)"
                ):
                    SummaryWriterContext.add_custom_scalars_multilinechart(
                        ["c", "d"], category="cat", title="title"
                    )
                SummaryWriterContext.add_custom_scalars_multilinechart(
                    ["e", "f"], category="cat", title="title2"
                )
                SummaryWriterContext.add_custom_scalars_multilinechart(
                    ["g", "h"], category="cat2", title="title"
                )

            SummaryWriterContext.add_custom_scalars(writer)
            writer.add_custom_scalars.assert_called_once_with(
                {
                    "cat": {
                        "title": ["Multiline", ["a", "b"]],
                        "title2": ["Multiline", ["e", "f"]],
                    },
                    "cat2": {"title": ["Multiline", ["g", "h"]]},
                }
            )
