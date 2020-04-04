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

logger = logging.getLogger(__name__)


class TestPreprocessing(SQLTestCase):
    def getConf(self):
        from pyspark import SparkConf

        conf = SparkConf()
        # set shuffle partitions to a low number, e.g. <= cores * 2 to speed
        # things up, otherwise the tests will use the default 200 partitions
        # and it will take a lot more time to complete
        conf.set("spark.sql.shuffle.partitions", "12")
        return conf

    def test_preprocessing(self):
        distributions = {}
        distributions["0"] = {"mean": 0, "stddev": 1, "size": (5,)}
        distributions["1"] = {"mean": 4, "stddev": 3, "size": (3,)}

        def get_random_feature():
            return {
                k: np.random.normal(
                    loc=info["mean"], scale=info["stddev"], size=info["size"]
                ).tolist()
                for k, info in distributions.items()
            }

        np.random.seed(42)
        data = [(i, get_random_feature()) for i in range(100000)]
        df = self.sc.parallelize(data).toDF(["i", "states"])
        df.show()

        table_name = "test_table"
        df.createOrReplaceTempView(table_name)

        num_samples = 10000
        preprocessing_options = PreprocessingOptions(num_samples=num_samples)

        table_spec = TableSpec(table_name=table_name)

        normalization_params = identify_normalization_parameters(
            table_spec, "states", preprocessing_options, seed=42
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
