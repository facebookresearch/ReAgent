#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call

import torch
from ml.rl.tensorboardX import SummaryWriterContext, summary_writer_context
from tensorboardX import SummaryWriter


class TestSummaryWriterContext(unittest.TestCase):
    def setUp(self):
        SummaryWriterContext._reset_globals()

    def tearDown(self):
        SummaryWriterContext._reset_globals()

    def test_noop(self):
        self.assertIsNone(SummaryWriterContext.add_scalar("test", torch.ones(1)))

    def test_with_none(self):
        with summary_writer_context(None):
            self.assertIsNone(SummaryWriterContext.add_scalar("test", torch.ones(1)))

    def test_writing(self):
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock()
            with summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))
            writer.add_scalar.assert_called_once_with(
                "test", torch.ones(1), global_step=0
            )

    def test_writing_stack(self):
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

    def test_swallowing_exception(self):
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock(side_effect=NotImplementedError("test"))
            writer.exceptions_to_ignore = (NotImplementedError, KeyError)
            with summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))

    def test_not_swallowing_exception(self):
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock(side_effect=NotImplementedError("test"))
            with self.assertRaisesRegex(
                NotImplementedError, "test"
            ), summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))

    def test_global_step(self):
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

    def test_add_custom_scalars(self):
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
