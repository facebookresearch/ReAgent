#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import random
import shutil

import numpy as np
import torch
from pyspark import SparkConf
from reagent.workflow.spark_utils import DEFAULT_SPARK_CONFIG
from sparktestingbase.sqltestcase import SQLTestCase


HIVE_METASTORE = "metastore_db"
SEED = 42


class ReagentSQLTestBase(SQLTestCase):
    def getConf(self):
        conf = SparkConf()
        for k, v in DEFAULT_SPARK_CONFIG.items():
            conf.set(k, v)
        return conf

    def setUp(self):
        super().setUp()
        assert not os.path.isdir(
            HIVE_METASTORE
        ), f"{HIVE_METASTORE} already exists! Try deleting it."

        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        logging.basicConfig()

    def tearDown(self):
        super().tearDown()

        # removes Derby from last runs
        if os.path.isdir(HIVE_METASTORE):
            shutil.rmtree(HIVE_METASTORE)
