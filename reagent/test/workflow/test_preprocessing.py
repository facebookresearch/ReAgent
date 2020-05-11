#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import numpy as np
import pytest
from reagent.preprocessing.identify_types import CONTINUOUS
from reagent.test.workflow.reagent_sql_test_base import ReagentSQLTestBase
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.types import PreprocessingOptions, TableSpec


logger = logging.getLogger(__name__)

NUM_ROWS = 10000
COL_NAME = "states"
TABLE_NAME = "test_table"


class TestPreprocessing(ReagentSQLTestBase):
    def setUp(self):
        super().setUp()
        logging.getLogger(__name__).setLevel(logging.INFO)

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
        df = self.sc.parallelize(data).toDF(["i", COL_NAME])
        df.show()

        df.createOrReplaceTempView(TABLE_NAME)

        num_samples = NUM_ROWS // 2
        preprocessing_options = PreprocessingOptions(num_samples=num_samples)

        table_spec = TableSpec(table_name=TABLE_NAME)

        normalization_params = identify_normalization_parameters(
            table_spec, COL_NAME, preprocessing_options, seed=self.test_class_seed
        )

        logger.info(normalization_params)
        for k, info in distributions.items():
            logger.info(
                f"Expect {k} to be normal with "
                f"mean {info['mean']}, stddev {info['stddev']}."
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
