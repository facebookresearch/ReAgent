""" Saves RLTimelineOperator's output as Petastorm parquet files.

Uses PySpark and Petastorm to convert RLTimelineOperator's output (in Hive) to
parquet files. A key functionality is that each feature in the form of a Map
(e.g. state_features), which Arrow can't yet handle, are converted to a dense
array and a presence array, which is 1 iff ith feature_id is present.

Additional transformations include: 
- action, which was originally a string, will be represented by the index in
the given list of possible actions
- possible_actions, which was originally a list of strings, will be represented
as a bitvector mask of length len(actions).
"""
import pyspark
import petastorm
from os.path import expanduser, join, abspath
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from pyspark.sql.functions import udf, struct, rand
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from pyspark.sql.types import (
    StructType,
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    DoubleType,
    MapType,
)
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
import numpy as np
from petastorm.pytorch import DataLoader

from typing import List, Dict, Any
from ml.rl.workflow.helpers import parse_args
import logging
import sys

logger = logging.getLogger(__name__)

# This is the type we use to store one-hot indices
ONE_HOT_TYPE = np.int64


def pyspark_to_numpy_types(pyspark_type):
    """ 
    Converts pyspark.sql.types to their numpy equivalent.
    """
    if isinstance(pyspark_type, DoubleType):
        return np.float64
    elif isinstance(pyspark_type, IntegerType):
        return np.int32
    elif isinstance(pyspark_type, LongType):
        return np.int64
    else:
        raise NotImplementedError(
            f"PySpark type {pyspark_type} does not have numpy equivalent."
        )


def get_petastorm_schema(
    actions: List[str], feature_map: Dict[str, List[Any]], sparse_pyspark_schema
) -> Unischema:
    """
    Creates the Petastorm storage schema from the schema of the
    RLTimelineOperator's sparse output. 

    Args:
        actions: List of possible actions
        feature_map: Some features (e.g. state_features) are mappings of 
            feature_id to value. This argument is a mapping of such features to
            the possible feature_ids that it may have. 
        sparse_pyspark_schema: Schema of RLTimelineOperator's output when loaded
            directly by PySpark.

    Returns:
        Petastorm storage schema.
    """
    unischema_fields = []

    def add_field(name, petastorm_type, shape, codec):
        # nothing can be null
        unischema_fields.append(
            UnischemaField(name, petastorm_type, shape, codec, False)
        )

    for struct_field in sparse_pyspark_schema:
        # first handle actions and action masks
        if struct_field.name in ["action", "next_action"]:
            add_field(
                name=struct_field.name,
                petastorm_type=ONE_HOT_TYPE,
                shape=(),
                codec=ScalarCodec(LongType()),
            )
        elif struct_field.name in ["possible_actions", "possible_next_actions"]:
            add_field(
                name=struct_field.name,
                petastorm_type=ONE_HOT_TYPE,
                shape=(len(actions),),
                codec=NdarrayCodec(),
            )
        elif struct_field.name in ["mdp_id"]:
            # mdp_id string will be hashed into a scalar ID
            add_field(
                name=struct_field.name,
                petastorm_type=ONE_HOT_TYPE,
                shape=(),
                codec=ScalarCodec(LongType()),
            )
        # now perform sparse2dense
        elif isinstance(struct_field.dataType, MapType):
            val_type = struct_field.dataType.valueType
            assert not isinstance(
                val_type, MapType
            ), f"{struct_field.name} has Map type with value type of Map"
            # add presence array
            add_field(
                name=f"{struct_field.name}_presence",
                petastorm_type=np.int64,
                shape=(len(feature_map[struct_field.name]),),
                codec=NdarrayCodec(),
            )
            # add dense array
            # also assume that mapped values are scalars
            add_field(
                name=struct_field.name,
                petastorm_type=pyspark_to_numpy_types(val_type),
                shape=(len(feature_map[struct_field.name]),),
                codec=NdarrayCodec(),
            )
        else:
            assert not isinstance(
                struct_field.dataType, ArrayType
            ), f"{struct_field.name} has array type"
            # simply add scalar field
            add_field(
                name=struct_field.name,
                petastorm_type=pyspark_to_numpy_types(struct_field.dataType),
                shape=(),
                codec=ScalarCodec(struct_field.dataType),
            )

    return Unischema("TimelineSchema", unischema_fields)


def preprocessing(
    actions: List[str], feature_map: Dict[str, List[Any]], petastorm_schema
):
    """ 
    Returns an RDD mapping function (mapping function on Rows) that converts 
    the RLTimelineOperator's output into the petastorm format, with actions
    one-hot encoded, with sparse features converted to dense, and with presence
    arrays.

    Args:
        actions: same as get_petastorm_schema
        feature_map: same as get_petastorm_schema
        petastorm_schema: Desired schema of the petastorm dataframe. 
    """

    def get_schema_type(feature_name):
        """ Returns the numpy dtype associated with the feature. """
        assert hasattr(
            petastorm_schema, feature_name
        ), f"{feature_name} does not exist."
        return getattr(petastorm_schema, feature_name).numpy_dtype

    def find_action_idx(desired_action):
        """ Returns the index of action in the list of possible actions. 
        If not found in the list of possible actions, return the number of
        possible actions.
        """
        for i, a in enumerate(actions):
            if a == desired_action:
                return i
        return len(actions)

    def row_map(row):
        """ The RDD mapping function """
        # convert Row to dict
        row_dict = row.asDict()

        # first handle one-hot encoding of action/masks
        action_keys = ["action", "next_action"]
        for k in action_keys:
            continue
            # row_dict[k] = find_action_idx(row_dict[k])

        possible_action_keys = ["possible_actions", "possible_next_actions"]
        for k in possible_action_keys:
            row_dict[k] = np.array(row_dict[k])
            # mask = np.zeros(len(actions), dtype=ONE_HOT_TYPE)
            # for a in row_dict[k]:
            #     i = find_action_idx(a)
            #     assert i < len(actions), f"action {a} (type {type(a)}) was not found in {actions}"
            #     mask[i] = 1
            # row_dict[k] = mask

        row_dict["mdp_id"] = hash(row_dict["mdp_id"])

        # now handle rest of keys, including converting sparse to dense
        used_keys = set(action_keys + possible_action_keys + ["mdp_id"])
        rest_features = row_dict.keys() - used_keys
        for feature in rest_features:
            val_type = get_schema_type(feature)
            val = row_dict[feature]
            # convert sparse to dense
            if isinstance(val, dict):
                presence_arr = []
                dense_arr = []
                # for every possible feature_id, check if it is part of
                # the sparse representation. If it is, mark it as present and
                # record it in the dense array. If it isn't, mark it
                # as absent and record a default value in dense array.
                for feature_id in feature_map[feature]:
                    if feature_id in val:
                        presence_arr.append(1)
                        dense_arr.append(val[feature_id])
                    else:
                        presence_arr.append(0)
                        # we assume values are a number here
                        dense_arr.append(0.0)
                presence_key = f"{feature}_presence"
                row_dict[presence_key] = np.array(
                    presence_arr, dtype=get_schema_type(presence_key)
                )
                row_dict[feature] = np.array(dense_arr, dtype=val_type)
            # if not a sparse map, simply copy it
            else:
                assert not isinstance(val, list)
                row_dict[feature] = val
        return dict_to_spark_row(petastorm_schema, row_dict)

    return row_map


def get_spark_session(warehouse_dir="spark-warehouse"):
    warehouse_dir = abspath(warehouse_dir)
    spark = (
        pyspark.sql.SparkSession.builder.master("local[1]")
        .enableHiveSupport()
        .getOrCreate()
    )
    return spark


def save_petastorm_dataset(
    actions,
    feature_map,
    input_table,
    output_table,
    shuffle=False,
    seed=42,
    row_group_size_mb=256,
):
    """ Assuming RLTimelineOperator output is in Hive storage at warehouse_dir,
        save the petastorm dataset after preprocessing.

        Args:
            actions: same as get_petastorm_schema
            feature_map: same as get_petastorm_schema
            input_table: table to read output of RLTimelineOperator
            output_table: location to store the dataset
            warehouse_dir: location where RLTimelineOperator stored data
            row_group_size_mb: parquet row group size for dataset
    """
    spark = get_spark_session()
    output_uri = f"file://{abspath(output_table)}"

    df = spark.read.table(input_table)
    peta_schema = get_petastorm_schema(actions, feature_map, df.schema)
    with materialize_dataset(spark, output_uri, peta_schema, row_group_size_mb):
        if shuffle:
            df = df.orderBy(rand(seed))
        rdd = df.rdd.map(preprocessing(actions, feature_map, peta_schema))
        peta_df = spark.createDataFrame(rdd, peta_schema.as_spark_schema())
        peta_df.write.mode("overwrite").parquet(output_uri)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    # hardcoded paths for dqn workflow
    train_input_table = "cartpole_discrete_training"
    train_output_table = "training_data/cartpole_discrete_timeline"
    eval_input_table = "cartpole_discrete_eval"
    eval_output_table = "training_data/cartpole_discrete_timeline_eval"

    # list of all the actions (for one-hot encoding)
    # NOTE: no longer needed
    actions = [0, 1]

    # list of all the features and their possible values
    feature_map = {
        "state_features": [0, 1, 2, 3],
        "next_state_features": [0, 1, 2, 3],
        "metrics": ["reward"],
    }

    save_petastorm_dataset(
        actions, feature_map, train_input_table, train_output_table, shuffle=True
    )
    logger.info(f"Saved training table as {train_output_table}")

    save_petastorm_dataset(
        actions, feature_map, eval_input_table, eval_output_table, shuffle=False
    )
    logger.info(f"Saved training table as {eval_output_table}")
