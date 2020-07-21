#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import numpy as np

# pyre-fixme[21]: Could not find `pytest`.
import pytest

# pyre-fixme[21]: Could not find `pyspark`.
from pyspark.sql.functions import asc

# pyre-fixme[21]: Could not find `workflow`.
from reagent.test.workflow.reagent_sql_test_base import ReagentSQLTestBase
from reagent.test.workflow.test_data.ex_mdps import generate_discrete_mdp_pandas_df
from reagent.workflow.data_fetcher import query_data
from reagent.workflow.types import Dataset, TableSpec


logger = logging.getLogger(__name__)


def generate_data_discrete(sqlCtx, multi_steps: bool, table_name: str):
    # pyre-fixme[16]: Module `test` has no attribute `workflow`.
    df, _ = generate_discrete_mdp_pandas_df(
        multi_steps=multi_steps, use_seq_num_diff_as_time_diff=False
    )
    df = sqlCtx.createDataFrame(df)
    logger.info("Created dataframe")
    df.show()
    df.createOrReplaceTempView(table_name)


# pyre-fixme[11]: Annotation `ReagentSQLTestBase` is not defined as a type.
class TestQueryData(ReagentSQLTestBase):
    def setUp(self):
        super().setUp()
        logging.getLogger(__name__).setLevel(logging.INFO)
        self.table_name = "test_table"
        logger.info(f"Table name is {self.table_name}")

    def generate_data(self, multi_steps=False):
        generate_data_discrete(
            self.sqlCtx, multi_steps=multi_steps, table_name=self.table_name
        )

    def _discrete_read_data(
        self, custom_reward_expression=None, gamma=None, multi_steps=None
    ):
        ts = TableSpec(table_name=self.table_name)
        dataset: Dataset = query_data(
            input_table_spec=ts,
            discrete_action=True,
            actions=["L", "R", "U", "D"],
            custom_reward_expression=custom_reward_expression,
            multi_steps=multi_steps,
            gamma=gamma,
        )
        df = self.sqlCtx.read.parquet(dataset.parquet_url)
        df = df.orderBy(asc("sequence_number"))
        logger.info("Read parquet dataframe: ")
        df.show()
        return df

    @pytest.mark.serial
    def test_query_data(self):
        # single step
        self.generate_data()
        df = self._discrete_read_data()
        df = df.toPandas()
        self.verify_discrete_single_step_except_rewards(df)
        self.assertEq(df["reward"], np.array([0.0, 1.0, 4.0, 5.0], dtype="float32"))
        logger.info("discrete single-step seems fine")

        # single step with reward := reward^3 + 10
        df = self._discrete_read_data(custom_reward_expression="POWER(reward, 3) + 10")
        df = df.toPandas()
        self.verify_discrete_single_step_except_rewards(df)
        self.assertEq(
            df["reward"], np.array([10.0, 11.0, 74.0, 135.0], dtype="float32")
        )
        logger.info("discrete single-step custom reward seems fine")

        # multi-step
        gamma = 0.9
        self.generate_data(multi_steps=True)
        df = self._discrete_read_data(multi_steps=2, gamma=gamma)
        df = df.toPandas()
        self.verify_discrete_multi_steps_except_rewards(df)
        self.assertAllClose(
            df["reward"],
            np.array(
                [gamma * 1, 1 * 1.0 + gamma * 4, 1 * 4.0 + gamma * 5, 1 * 5.0],
                dtype="float32",
            ),
        )
        logger.info("discrete multi-step seems fine.")

    def verify_discrete_single_step_except_rewards(self, df):
        """ expects a pandas dataframe """
        self.assertEq(df["sequence_number"], np.array([1, 2, 3, 4], dtype="int32"))

        state_features_presence = np.array(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
            ],
            dtype="bool",
        )
        self.assertEq(df["state_features_presence"], state_features_presence)
        state_features = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype="float32",
        )
        self.assertEqWithPresence(
            df["state_features"], state_features_presence, state_features
        )

        self.assertEq(df["action"], np.array([0, 1, 2, 3]))
        self.assertEq(
            df["action_probability"], np.array([0.3, 0.4, 0.5, 0.6], dtype="float32")
        )
        self.assertEq(df["not_terminal"], np.array([1, 1, 1, 0], dtype="bool"))
        next_state_features_presence = np.array(
            [
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            dtype="bool",
        )
        self.assertEq(df["next_state_features_presence"], next_state_features_presence)
        next_state_features = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype="float32",
        )
        self.assertEqWithPresence(
            df["next_state_features"], next_state_features_presence, next_state_features
        )

        self.assertEq(df["next_action"], np.array([1, 2, 3, 4]))
        self.assertEq(df["time_diff"], np.array([1, 3, 1, 1]))
        self.assertEq(df["step"], np.array([1, 1, 1, 1]))
        self.assertEq(
            df["possible_actions_mask"],
            np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
        )
        self.assertEq(
            df["possible_next_actions_mask"],
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]),
        )

    def verify_discrete_multi_steps_except_rewards(self, df):
        self.assertEq(df["sequence_number"], np.array([1, 2, 3, 4], dtype="int32"))

        state_features_presence = np.array(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
            ],
            dtype="bool",
        )
        self.assertEq(df["state_features_presence"], state_features_presence)
        state_features = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype="float32",
        )
        self.assertEqWithPresence(
            df["state_features"], state_features_presence, state_features
        )

        self.assertEq(df["action"], np.array([0, 1, 2, 3]))
        self.assertEq(
            df["action_probability"], np.array([0.3, 0.4, 0.5, 0.6], dtype="float32")
        )
        self.assertEq(df["not_terminal"], np.array([1, 1, 0, 0], dtype="bool"))

        next_state_features_presence = np.array(
            [
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
                [False, False, False, False, True],
            ],
            dtype="bool",
        )
        self.assertEq(df["next_state_features_presence"], next_state_features_presence)
        next_state_features = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype="float32",
        )
        self.assertEqWithPresence(
            df["next_state_features"], next_state_features_presence, next_state_features
        )

        self.assertEq(df["next_action"], np.array([2, 3, 4, 4]))
        self.assertEq(df["time_diff"], np.array([1, 1, 1, 1]))
        self.assertEq(df["step"], np.array([2, 2, 2, 1]))
        self.assertEq(
            df["possible_actions_mask"],
            np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
        )
        self.assertEq(
            df["possible_next_actions_mask"],
            np.array([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        )


if __name__ == "__main__":
    unittest.main()
