#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import torch
from ml.rl.tensorboardX import SummaryWriterContext, summary_writer_context
from tensorboardX import SummaryWriter


class TestSummaryWriterContext(unittest.TestCase):
    def test_noop(self):
        self.assertIsNone(SummaryWriterContext.add_scalar("test", torch.ones(1)))

    def test_writing(self):
        with TemporaryDirectory() as tmp_dir:
            writer = SummaryWriter(tmp_dir)
            writer.add_scalar = MagicMock()
            with summary_writer_context(writer):
                SummaryWriterContext.add_scalar("test", torch.ones(1))
            writer.add_scalar.assert_called_once_with("test", torch.ones(1))

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
            writer1.add_scalar.assert_called_once_with("test1", torch.zeros(1))
            writer2.add_scalar.assert_called_once_with("test2", torch.ones(1))
