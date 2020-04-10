#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
import numpy as np
from reagent.preprocessing.identify_types import CONTINUOUS
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.types import PreprocessingOptions, TableSpec
from sparktestingbase.sqltestcase import SQLTestCase
import logging
import pytest

logger = logging.getLogger(__name__)

SEED = 42
NUM_ROWS = 10000


class TestPreprocessing(SQLTestCase):
    def getConf(self):
        from pyspark import SparkConf

        conf = SparkConf()
        # set shuffle partitions to a low number, e.g. <= cores * 2 to speed
        # things up, otherwise the tests will use the default 200 partitions
        # and it will take a lot more time to complete
        conf.set("spark.sql.shuffle.partitions", "12")
        conf.set("spark.sql.warehouse.dir", "_test_preprocessing_warehouse")
        conf.set("spark.port.maxRetries", "30")
        return conf

    def setUp(self):
        import os, shutil

        if os.path.isdir("metastore_db"):
            shutil.rmtree("metastore_db")

        np.random.seed(SEED)
        logging.basicConfig()
        logging.getLogger(__name__).setLevel(logging.INFO)
        super().setUp()

    @pytest.mark.serial
    def test_preprocessing(self):
        distributions = {}
        distributions["0"] = {"mean": 0, "stddev": 1}
        distributions["1"] = {"mean": 4, "stddev": 3}

        def get_random_feature():
            return {
                k: np.random.normal(loc=info["mean"], scale=info["stddev"])
                for k, info in distributions.items()
            }

        data = [(i, get_random_feature()) for i in range(NUM_ROWS)]
        df = self.sc.parallelize(data).toDF(["i", "states"])
        df.show()

        table_name = "test_table"
        df.createOrReplaceTempView(table_name)

        num_samples = NUM_ROWS / 2
        preprocessing_options = PreprocessingOptions(num_samples=num_samples)

        table_spec = TableSpec(table_name=table_name)

        normalization_params = identify_normalization_parameters(
            table_spec, "states", preprocessing_options, seed=SEED
        )

        logger.info(normalization_params)
        for k, info in distributions.items():
            logger.info(
                f"Expect {k} to be normal with mean {info['mean']}, stddev {info['stddev']}"
            )
            assert normalization_params[k].feature_type == CONTINUOUS
            assert (
                abs(normalization_params[k].mean - info["mean"]) < 0.05
            ), f"{normalization_params[k].mean} not close to {info['mean']}"
            assert abs(
                normalization_params[k].stddev - info["stddev"] < 0.2
            ), f"{normalization_params[k].stddev} not close to {info['stddev']}"
        logger.info("identify_normalization_parameters seems fine.")


if __name__ == "__main__":
    unittest.main()
