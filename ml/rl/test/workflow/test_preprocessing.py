#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import unittest

import torch
from ml.rl.prediction.dqn_torch_predictor import (
    DiscreteDqnTorchPredictor,
    ParametricDqnTorchPredictor,
)
from ml.rl.test.base.horizon_test_base import HorizonTestBase
from ml.rl.workflow.identify_types_flow import (
    get_spark_session,
    identify_normalization_parameters,
)
from ml.rl.preprocessing.identify_types import CONTINUOUS
import numpy as np

from ml.rl.workflow.types import TableSpec, PreprocessingOptions


class TestPreprocessing(HorizonTestBase):
    def test_preprocessing(self):
        spark = get_spark_session()

        distributions = dict()
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
        df = spark.sparkContext.parallelize(data).toDF(["i", "states"])
        df.show()

        table_name = "test_table"
        df.createOrReplaceTempView(table_name)

        num_samples = 10000
        preprocessing_options = PreprocessingOptions(num_samples=num_samples)

        table_spec = TableSpec(table_name=table_name)

        normalization_params = identify_normalization_parameters(
            table_spec, "states", preprocessing_options, seed=42
        )

        print(normalization_params)
        for k, info in distributions.items():
            print(
                f"Expect {k} to be normal with mean {info['mean']}, stddev {info['stddev']}"
            )
            assert normalization_params[k].feature_type == CONTINUOUS
            assert (
                abs(normalization_params[k].mean - info["mean"]) < 0.05
            ), f"{normalization_params[k].mean} not close to {info['mean']}"
            assert abs(
                normalization_params[k].stddev - info["stddev"] < 0.2
            ), f"{normalization_params[k].stddev} not close to {info['stddev']}"
        print("Everything seems fine.")

        spark.stop()


if __name__ == "__main__":
    unittest.main()
