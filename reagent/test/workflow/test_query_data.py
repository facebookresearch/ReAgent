#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import tempfile
import unittest
from os.path import abspath

import numpy as np
import pandas
import pytest
from pyspark.sql.functions import asc
from reagent.test.environment.environment import MultiStepSamples
from reagent.test.workflow.reagent_sql_test_base import ReagentSQLTestBase
from reagent.workflow.data_fetcher import query_data
from reagent.workflow.types import Dataset, TableSpec


logger = logging.getLogger(__name__)


def generate_data_discrete(sqlCtx, multi_steps: bool, table_name: str):
    # Simulate the following MDP:
    # state: 0, action: 7 ('L'), reward: 0,
    # state: 1, action: 8 ('R'), reward: 1,
    # state: 4, action: 9 ('U'), reward: 4,
    # state: 5, action: 10 ('D'), reward: 5,
    # state: 6 (terminal)
    actions = ["L", "R", "U", "D"]
    possible_actions = [["L", "R"], ["R", "U"], ["U", "D"], ["D"]]

    # assume multi_steps=2
    if multi_steps:
        rewards = [[0, 1], [1, 4], [4, 5], [5]]
        metrics = [
            [{"reward": 0}, {"reward": 1}],
            [{"reward": 1}, {"reward": 4}],
            [{"reward": 4}, {"reward": 5}],
            [{"reward": 5}],
        ]
        next_states = [[{1: 1}, {4: 1}], [{4: 1}, {5: 1}], [{5: 1}, {6: 1}], [{6: 1}]]
        next_actions = [["R", "U"], ["U", "D"], ["D", ""], [""]]
        possible_next_actions = [
            [["R", "U"], ["U", "D"]],
            [["U", "D"], ["D"]],
            [["D"], [""]],
            [[""]],
        ]
        terminals = [[0, 0], [0, 0], [0, 1], [1]]
        time_diffs = [[1, 1], [1, 1], [1, 1], [1]]
    else:
        rewards = [[0], [1], [4], [5]]
        metrics = [{"reward": 0}, {"reward": 1}, {"reward": 4}, {"reward": 5}]  # noqa
        next_states = [[{1: 1}], [{4: 1}], [{5: 1}], [{6: 1}]]
        next_actions = [["R"], ["U"], ["D"], [""]]
        possible_next_actions = [[["R", "U"]], [["U", "D"]], [["D"]], [[""]]]
        terminals = [[0], [0], [0], [1]]
        time_diffs = [1, 3, 1, 1]  # noqa

    samples = MultiStepSamples(
        mdp_ids=["0", "0", "0", "0"],
        sequence_numbers=[0, 1, 4, 5],
        sequence_number_ordinals=[1, 2, 3, 4],
        states=[{0: 1}, {1: 1}, {4: 1}, {5: 1}],
        actions=actions,
        action_probabilities=[0.3, 0.4, 0.5, 0.6],
        rewards=rewards,
        possible_actions=possible_actions,
        next_states=next_states,
        next_actions=next_actions,
        terminals=terminals,
        possible_next_actions=possible_next_actions,
    )
    if not multi_steps:
        samples = samples.to_single_step()

    next_state_features = samples.next_states
    possible_next_actions = samples.possible_next_actions
    next_actions = samples.next_actions

    df = pandas.DataFrame(
        {
            "mdp_id": samples.mdp_ids,
            "sequence_number": samples.sequence_numbers,
            "sequence_number_ordinal": samples.sequence_number_ordinals,
            "state_features": samples.states,
            "action": samples.actions,
            "action_probability": samples.action_probabilities,
            "reward": samples.rewards,
            "next_state_features": next_state_features,
            "next_action": next_actions,
            "time_diff": time_diffs,
            "possible_actions": samples.possible_actions,
            "possible_next_actions": possible_next_actions,
            "metrics": metrics,
        }
    )
    df = sqlCtx.createDataFrame(df)
    logger.info("Created dataframe")
    df.show()
    df.createOrReplaceTempView(table_name)


def assertEq(series_a, arr_b):
    arr_a = np.array(series_a.tolist())
    np.testing.assert_equal(arr_a, arr_b)


def assertAllClose(series_a, arr_b):
    arr_a = np.array(series_a.tolist())
    np.testing.assert_allclose(arr_a, arr_b)


def assertEqWithPresence(series_a, presence, arr_b):
    arr_a = np.array(series_a.tolist())
    present_a = arr_a[presence]
    present_b = arr_b[presence]
    np.testing.assert_equal(present_a, present_b)


def verify_single_step_except_rewards(df):
    """ expects a pandas dataframe """
    assertEq(df["sequence_number"], pandas.Series([1, 2, 3, 4]))

    state_features_presence = np.array(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
        ],
        dtype="bool",
    )
    assertEq(df["state_features_presence"], state_features_presence)
    state_features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    assertEqWithPresence(df["state_features"], state_features_presence, state_features)

    assertEq(df["action"], np.array([0, 1, 2, 3]))
    assertEq(df["action_probability"], np.array([0.3, 0.4, 0.5, 0.6], dtype="float32"))
    assertEq(df["not_terminal"], np.array([1, 1, 1, 0], dtype="bool"))
    next_state_features_presence = np.array(
        [
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
        ],
        dtype="bool",
    )
    assertEq(df["next_state_features_presence"], next_state_features_presence)
    next_state_features = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    assertEqWithPresence(
        df["next_state_features"], next_state_features_presence, next_state_features
    )

    assertEq(df["next_action"], np.array([1, 2, 3, 4]))
    assertEq(df["time_diff"], np.array([1, 3, 1, 1]))
    assertEq(df["step"], np.array([1, 1, 1, 1]))
    assertEq(
        df["possible_actions_mask"],
        np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
    )
    assertEq(
        df["possible_next_actions_mask"],
        np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]),
    )


def verify_multi_steps_except_rewards(df):
    assertEq(df["sequence_number"], pandas.Series([1, 2, 3, 4]))

    state_features_presence = np.array(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
        ],
        dtype="bool",
    )
    assertEq(df["state_features_presence"], state_features_presence)
    state_features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    assertEqWithPresence(df["state_features"], state_features_presence, state_features)

    assertEq(df["action"], np.array([0, 1, 2, 3]))
    assertEq(df["action_probability"], np.array([0.3, 0.4, 0.5, 0.6], dtype="float32"))
    assertEq(df["not_terminal"], np.array([1, 1, 0, 0], dtype="bool"))

    next_state_features_presence = np.array(
        [
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
            [False, False, False, False, True],
        ],
        dtype="bool",
    )
    assertEq(df["next_state_features_presence"], next_state_features_presence)
    next_state_features = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    assertEqWithPresence(
        df["next_state_features"], next_state_features_presence, next_state_features
    )

    assertEq(df["next_action"], np.array([2, 3, 4, 4]))
    assertEq(df["time_diff"], np.array([1, 1, 1, 1]))
    assertEq(df["step"], np.array([2, 2, 2, 1]))
    assertEq(
        df["possible_actions_mask"],
        np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
    )
    assertEq(
        df["possible_next_actions_mask"],
        np.array([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )


def rand_string(length=10):
    import string
    import random

    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


class TestQueryData(ReagentSQLTestBase):
    def setUp(self):
        super().setUp()
        logging.getLogger(__name__).setLevel(logging.INFO)
        self.table_name = rand_string()
        self.temp_parquet = tempfile.TemporaryDirectory()
        self.parquet_url = f"file://{abspath(self.temp_parquet.name)}"
        logger.info(f"Table name is {self.table_name}")

    def tearDown(self):
        self.temp_parquet.cleanup()
        super().tearDown()

    def generate_data(self, multi_steps=False):
        generate_data_discrete(
            self.sqlCtx, multi_steps=multi_steps, table_name=self.table_name
        )

    def _read_data(self, custom_reward_expression=None, gamma=None, multi_steps=None):
        ts = TableSpec(
            table_name=self.table_name,
            output_dataset=Dataset(parquet_url=self.parquet_url),
        )
        query_data(
            input_table_spec=ts,
            states=[0, 1, 4, 5, 6],
            actions=["L", "R", "U", "D"],
            metrics=["reward"],
            custom_reward_expression=custom_reward_expression,
            multi_steps=multi_steps,
            gamma=gamma,
        )
        df = self.sqlCtx.read.parquet(self.parquet_url)
        df = df.orderBy(asc("sequence_number"))
        logger.info("Read parquet dataframe")
        df.show()
        return df

    @pytest.mark.serial
    def test_query_data_single_step(self):
        self.generate_data()
        df = self._read_data()
        df = df.toPandas()
        verify_single_step_except_rewards(df)
        assertEq(df["reward"], np.array([0.0, 1.0, 4.0, 5.0], dtype="float32"))
        logger.info("single-step seems fine")

    @pytest.mark.serial
    def test_query_data_single_step_custom_reward(self):
        self.generate_data()
        df = self._read_data(custom_reward_expression="POWER(reward, 3) + 10")
        df = df.toPandas()
        verify_single_step_except_rewards(df)
        assertEq(df["reward"], np.array([10.0, 11.0, 74.0, 135.0], dtype="float32"))
        logger.info("single-step custom reward seems fine")

    @pytest.mark.serial
    def test_query_data_multi_steps(self):
        gamma = 0.9
        self.generate_data(multi_steps=True)
        df = self._read_data(multi_steps=2, gamma=gamma)
        df = df.toPandas()
        verify_multi_steps_except_rewards(df)
        assertAllClose(
            df["reward"],
            np.array(
                [gamma * 1, 1 * 1.0 + gamma * 4, 1 * 4.0 + gamma * 5, 1 * 5.0],
                dtype="float32",
            ),
        )
        logger.info("multi-step seems fine.")


if __name__ == "__main__":
    unittest.main()
