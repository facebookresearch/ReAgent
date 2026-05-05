#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-strict

import logging
import unittest

import numpy as np
import pytest
from reagent.preprocessing.identify_types import CONTINUOUS
from reagent.test.oss_workflow.reagent_sql_test_base import ReagentSQLTestBase
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.types import PreprocessingOptions, TableSpec


logger = logging.getLogger(__name__)

NUM_ROWS = 10000
COL_NAME = "states"
TABLE_NAME = "test_table"


# pyre-fixme[11]: Annotation `ReagentSQLTestBase` is not defined as a type.
class TestPreprocessing(ReagentSQLTestBase):
    def setUp(self) -> None:
        super().setUp()
        logging.getLogger(__name__).setLevel(logging.INFO)

    @pytest.mark.serial
    def test_preprocessing(self) -> None:
        distributions = {}
        distributions["0"] = {"mean": 0, "stddev": 1}
        distributions["1"] = {"mean": 4, "stddev": 3}

        def get_random_feature() -> dict[str, float]:
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

        # pyrefly: ignore [missing-argument, unexpected-keyword]
        table_spec = TableSpec(table_name=TABLE_NAME)

        normalization_params = identify_normalization_parameters(
            # pyrefly: ignore [unexpected-keyword]
            table_spec,
            COL_NAME,
            preprocessing_options,
            # pyrefly: ignore [unexpected-keyword]
            seed=self.test_class_seed,
        )

        logger.info(normalization_params)
        for k, info in distributions.items():
            logger.info(
                f"Expect {k} to be normal with "
                f"mean {info['mean']}, stddev {info['stddev']}."
            )
            # pyrefly: ignore [bad-index]
            assert normalization_params[k].feature_type == CONTINUOUS
            # pyrefly: ignore [bad-index, unsupported-operation]
            assert abs(normalization_params[k].mean - info["mean"]) < 0.05, (
                # pyrefly: ignore [bad-index]
                f"{normalization_params[k].mean} not close to {info['mean']}"
            )
            # pyrefly: ignore [bad-index, unsupported-operation]
            assert abs(normalization_params[k].stddev - info["stddev"] < 0.2), (
                # pyrefly: ignore [bad-index]
                f"{normalization_params[k].stddev} not close to {info['stddev']}"
            )
        logger.info("identify_normalization_parameters seems fine.")


if __name__ == "__main__":
    unittest.main()
