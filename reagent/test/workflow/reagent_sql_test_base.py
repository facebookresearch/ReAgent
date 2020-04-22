#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import shutil

import numpy as np
import torch
from pyspark import SparkConf
from sparktestingbase.sqltestcase import SQLTestCase


SEED = 42


class ReagentSQLTestBase(SQLTestCase):
    def getConf(self):
        conf = SparkConf()
        # set shuffle partitions to a low number, e.g. <= cores * 2 to speed
        # things up, otherwise the tests will use the default 200 partitions
        # and it will take a lot more time to complete
        conf.set("spark.sql.shuffle.partitions", "12")
        conf.set("spark.port.maxRetries", "30")
        return conf

    def setUp(self):
        super().setUp()
        assert not os.path.isdir("metastore_db"), "metastore_db already exists"

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logging.basicConfig()

    def tearDown(self):
        super().tearDown()

        # removes Derby from last runs
        if os.path.isdir("metastore_db"):
            shutil.rmtree("metastore_db")
