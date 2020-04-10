#!/usr/bin/env python3
from typing import List, Dict, Tuple, Optional

from pyspark.sql.types import (
    LongType,
    FloatType,
    MapType,
    ArrayType,
    StructType,
    StructField,
    BooleanType,
)
from pyspark.sql.functions import col, udf, crc32
from reagent.workflow.types import TableSpec, Dataset
import logging
from reagent.workflow.spark_utils import get_spark_session

logger = logging.getLogger(__name__)

# for normalizing crc32 output
MAX_UINT32 = 4294967295


def calc_custom_reward(sqlCtx, df, custom_reward_expression: str):
    temp_table_name = "_tmp_calc_reward_df"
    temp_reward_name = "_tmp_reward_col"
    df.createOrReplaceTempView(temp_table_name)
    df = sqlCtx.sql(
        f"SELECT *, CAST(COALESCE({custom_reward_expression}, 0) AS FLOAT) as {temp_reward_name} FROM {temp_table_name}"
    )
    return df.drop("reward").withColumnRenamed(temp_reward_name, "reward")


def calc_reward_multi_steps(sqlCtx, df, multi_steps: int, gamma: float):
    # computes r_0 + gamma * (r_1 + gamma * (r_2 + ... ))
    expr = f"AGGREGATE(REVERSE(reward), FLOAT(0), (s, x) -> FLOAT({gamma}) * s + x)"
    return calc_custom_reward(sqlCtx, df, expr)


def perform_preprocessing(
    sqlCtx,
    table_spec: TableSpec,
    states: List[int],
    actions: List[str],
    metrics: List[str],
    custom_reward_expression: Optional[str] = None,
    sample_range: Optional[Tuple[float, float]] = None,
    multi_steps: Optional[int] = None,
    gamma: Optional[float] = None,
):
    """ Perform preprocessing of given dataframe df.
    Preprocessing steps include calculating the reward,
    performing sparse-to-dense for mapped columns like state_features
    and metrics, and subsampling based on sample_range.
    If multi_steps is set (with gamma), then we assume multi_steps RL setting.
    """
    if sample_range:
        assert (
            0.0 <= sample_range[0]
            and sample_range[0] <= sample_range[1]
            and sample_range[1] <= 100.0
        ), f"{sample_range} is invalid."

    df = sqlCtx.sql(f"SELECT * FROM {table_spec.table_name}")

    # after this, reward column should be set to be the reward now
    if custom_reward_expression is not None:
        df = calc_custom_reward(sqlCtx, df, custom_reward_expression)
    elif multi_steps is not None:
        assert gamma is not None
        df = calc_reward_multi_steps(sqlCtx, df, multi_steps, gamma)
    # assume single step case reward is already a column

    def get_step(next_col):
        """ get step count """
        if multi_steps is not None:
            return min(len(next_col), multi_steps)
        else:
            return 1

    get_step_udf = udf(get_step, LongType())
    df = df.withColumn("step", get_step_udf("next_state_features"))

    def make_next_udf(return_type):
        """ return udf to get next item, provided item type """

        def get_next(next_col):
            """ generic function to get the next item """
            if multi_steps is not None:
                step = min(len(next_col), multi_steps)
                return next_col[step - 1]
            else:
                return next_col

        return udf(get_next, return_type)

    df = df.withColumn("time_diff", make_next_udf(LongType())("time_diff"))

    def make_sparse2dense(df, col_name: str, possible_keys: List):
        """ Given a list of possible keys, convert sparse map to dense array.
            In our example, both value_type is assumed to be a float.
        """
        output_type = StructType(
            [
                StructField("presence", ArrayType(BooleanType()), False),
                StructField("dense", ArrayType(FloatType()), False),
            ]
        )

        def sparse2dense(map_col):
            assert isinstance(
                map_col, dict
            ), f"{map_col} has type {type(map_col)} and is not a dict."
            presence = []
            dense = []
            for key in possible_keys:
                val = map_col.get(key, None)
                if val is not None:
                    presence.append(True)
                    dense.append(float(val))
                else:
                    presence.append(False)
                    dense.append(0.0)
            return presence, dense

        sparse2dense_udf = udf(sparse2dense, output_type)
        df = df.withColumn(col_name, sparse2dense_udf(col_name))
        df = df.withColumn(f"{col_name}_presence", col(f"{col_name}.presence"))
        df = df.withColumn(col_name, col(f"{col_name}.dense"))
        return df

    df = make_sparse2dense(df, "state_features", states)

    next_map_udf = make_next_udf(MapType(LongType(), FloatType()))
    df = df.withColumn("next_state_features", next_map_udf("next_state_features"))
    df = make_sparse2dense(df, "next_state_features", states)

    df = df.withColumn("metrics", next_map_udf("metrics"))
    df = make_sparse2dense(df, "metrics", metrics)

    def where(arr: List[str]):
        """ locate the index of item in arr, len(arr) if not found. """

        def find(item: str):
            for i, arr_item in enumerate(arr):
                if arr_item == item:
                    return i
            return len(arr)

        return find

    where_udf = udf(where(actions), LongType())
    df = df.withColumn("action", where_udf("action"))
    df = df.withColumn(
        "next_action", where_udf(make_next_udf(LongType())("next_action"))
    )

    def get_not_terminal(next_action):
        """ terminal state iff next_action is "" (i.e. onehot len(actions))"""
        return next_action < len(actions)

    get_not_terminal_udf = udf(get_not_terminal, BooleanType())
    df = df.withColumn("not_terminal", get_not_terminal_udf("next_action"))

    def onehot(arr: List[str]):
        """ one-hot encode elements of arr depending on their existence in target """

        def encode(target: List[str]):
            result = [0] * len(arr)
            for i, arr_item in enumerate(arr):
                if arr_item in target:
                    result[i] = 1
            return result

        return encode

    onehot_udf = udf(onehot(actions), ArrayType(LongType()))
    df = df.withColumn("possible_actions_mask", onehot_udf("possible_actions"))
    df = df.withColumn(
        "possible_next_actions_mask",
        onehot_udf(make_next_udf(ArrayType(LongType()))("possible_next_actions")),
    )

    # assuming use_seq_num_diff_as_time_diff = False for now
    df = df.withColumn("sequence_number", col("sequence_number_ordinal"))

    # crc32 is treated as a cryptographic hash with range [0, MAX_UINT32-1]
    # Note: we're assuming no collisions!
    df = df.withColumn("mdp_id", crc32(col("mdp_id")))
    if sample_range:
        lower_bound = sample_range[0] / 100.0 * MAX_UINT32
        upper_bound = sample_range[1] / 100.0 * MAX_UINT32
        df = df.filter((lower_bound <= col("mdp_id")) & (col("mdp_id") <= upper_bound))

    # select all the relevant columns and perform type conversions
    return df.select(
        col("reward").cast(FloatType()),
        col("state_features").cast(ArrayType(FloatType())),
        col("state_features_presence").cast(ArrayType(BooleanType())),
        col("next_state_features").cast(ArrayType(FloatType())),
        col("next_state_features_presence").cast(ArrayType(BooleanType())),
        col("action").cast(LongType()),
        col("action_probability").cast(FloatType()),
        col("not_terminal").cast(BooleanType()),
        col("next_action").cast(LongType()),
        col("possible_actions_mask").cast(ArrayType(LongType())),
        col("possible_next_actions_mask").cast(ArrayType(LongType())),
        col("mdp_id").cast(LongType()),
        col("sequence_number").cast(LongType()),
        col("step").cast(LongType()),
        col("time_diff").cast(LongType()),
        col("metrics").cast(ArrayType(FloatType())),
        col("metrics_presence").cast(ArrayType(BooleanType())),
    )


def query_data(
    input_table_spec: TableSpec,
    states: List[int],
    actions: List[str],
    metrics: List[str],
    custom_reward_expression: Optional[str] = None,
    sample_range: Optional[Tuple[float, float]] = None,
    multi_steps: Optional[int] = None,
    gamma: Optional[float] = None,
) -> None:
    sqlCtx = get_spark_session()
    # includes rewards preprocessing, sparse2dense
    preprocessed_df = perform_preprocessing(
        sqlCtx,
        table_spec=input_table_spec,
        states=states,
        actions=actions,
        metrics=metrics,
        custom_reward_expression=custom_reward_expression,
        sample_range=sample_range,
        multi_steps=multi_steps,
        gamma=gamma,
    )
    preprocessed_df.write.mode("overwrite").parquet(input_table_spec.output_dataset.parquet_url)
    return
