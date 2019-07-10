#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from ml.rl.tensorboardX import SummaryWriterContext


class HorizonTestBase(unittest.TestCase):
    def setUp(self):
        SummaryWriterContext._reset_globals()

    def tearDown(self):
        SummaryWriterContext._reset_globals()
