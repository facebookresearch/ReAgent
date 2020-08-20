#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

import reagent.core.types as rlt

# pyre-fixme[21]: Could not find `petastorm`.
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader, decimal_friendly_collate

# pyre-fixme[21]: Could not find `pyspark`.
from pyspark.sql.functions import col, crc32, explode, map_keys, udf

# pyre-fixme[21]: Could not find module `pyspark.sql.types`.
# pyre-fixme[21]: Could not find module `pyspark.sql.types`.
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    LongType,
    MapType,
    StructField,
    StructType,
)
from reagent.core.types import (
    Dataset,
    OssDataset,
    PreprocessingOptions,
    ReaderOptions,
    TableSpec,
)
from reagent.data_fetchers.data_fetcher import DataFetcher
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.parameters import NormalizationParameters
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.torch_utils import dict_to_tensor
from reagent.training import RLTrainer, SACTrainer, TD3Trainer
from reagent.workflow.identify_types_flow import identify_normalization_parameters
from reagent.workflow.spark_utils import get_spark_session, get_table_url


logger = logging.getLogger(__name__)

# for normalizing crc32 output
MAX_UINT32 = 4294967295

# for generating/checking random tmp table names for upload_as_parquet
UPLOAD_PARQUET_TMP_SUFFIX_LEN = 10
MAX_UPLOAD_PARQUET_TRIES = 10


def calc_custom_reward(df, custom_reward_expression: str):
    sqlCtx = get_spark_session()
    # create a temporary table for running sql
    temp_table_name = "_tmp_calc_reward_df"
    temp_reward_name = "_tmp_reward_col"
    df.createOrReplaceTempView(temp_table_name)
    df = sqlCtx.sql(
        f"SELECT *, CAST(COALESCE({custom_reward_expression}, 0) AS FLOAT)"
        f" as {temp_reward_name} FROM {temp_table_name}"
    )
    return df.drop("reward").withColumnRenamed(temp_reward_name, "reward")


def calc_reward_multi_steps(df, multi_steps: int, gamma: float):
    # assumes df[reward] is array[float] and 1 <= len(df[reward]) <= multi_steps
    # computes r_0 + gamma * (r_1 + gamma * (r_2 + ... ))
    expr = f"AGGREGATE(REVERSE(reward), FLOAT(0), (s, x) -> FLOAT({gamma}) * s + x)"
    return calc_custom_reward(df, expr)


def set_reward_col_as_reward(
    df,
    custom_reward_expression: Optional[str] = None,
    multi_steps: Optional[int] = None,
    gamma: Optional[float] = None,
):
    # after this, reward column should be set to be the reward
    if custom_reward_expression is not None:
        df = calc_custom_reward(df, custom_reward_expression)
    elif multi_steps is not None:
        assert gamma is not None
        df = calc_reward_multi_steps(df, multi_steps, gamma)
    return df


def hash_mdp_id_and_subsample(df, sample_range: Optional[Tuple[float, float]] = None):
    """ Since mdp_id is a string but Pytorch Tensors do not store strings,
    we hash them with crc32, which is treated as a cryptographic hash
    (with range [0, MAX_UINT32-1]). We also perform an optional subsampling
    based on this hash value.
    NOTE: we're assuming no collisions in this hash! Otherwise, two mdp_ids
    can be indistinguishable after the hash.
    TODO: change this to a deterministic subsample.
    """
    if sample_range:
        assert (
            0.0 <= sample_range[0]
            and sample_range[0] <= sample_range[1]
            and sample_range[1] <= 100.0
        ), f"{sample_range} is invalid."

    df = df.withColumn("mdp_id", crc32(col("mdp_id")))
    if sample_range:
        lower_bound = sample_range[0] / 100.0 * MAX_UINT32
        upper_bound = sample_range[1] / 100.0 * MAX_UINT32
        df = df.filter((lower_bound <= col("mdp_id")) & (col("mdp_id") <= upper_bound))
    return df


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


#################################################
# Below are some UDFs we use for preprocessing. #
#################################################


def make_get_step_udf(multi_steps: Optional[int]):
    """ Get step count by taking length of next_states_features array. """

    def get_step(col: List):
        return 1 if multi_steps is None else min(len(col), multi_steps)

    return udf(get_step, LongType())


def make_next_udf(multi_steps: Optional[int], return_type):
    """ Generic udf to get next (after multi_steps) item, provided item type. """

    def get_next(next_col):
        return (
            next_col
            if multi_steps is None
            else next_col[min(len(next_col), multi_steps) - 1]
        )

    return udf(get_next, return_type)


def make_where_udf(arr: List[str]):
    """ Return index of item in arr, and len(arr) if not found. """

    def find(item: str):
        for i, arr_item in enumerate(arr):
            if arr_item == item:
                return i
        return len(arr)

    return udf(find, LongType())


def make_existence_bitvector_udf(arr: List[str]):
    """ one-hot encode elements of target depending on their existence in arr. """

    default = [0] * len(arr)

    def encode(target: List[str]):
        bitvec = default.copy()
        for i, arr_item in enumerate(arr):
            if arr_item in target:
                bitvec[i] = 1
        return bitvec

    return udf(encode, ArrayType(LongType()))


def misc_column_preprocessing(df, multi_steps: Optional[int]):
    """ Miscellaneous columns are step, time_diff, sequence_number, not_terminal. """

    # step refers to n in n-step RL; special case when approaching terminal
    df = df.withColumn("step", make_get_step_udf(multi_steps)("next_state_features"))

    # take the next time_diff
    next_long_udf = make_next_udf(multi_steps, LongType())
    df = df.withColumn("time_diff", next_long_udf("time_diff"))

    # assuming use_seq_num_diff_as_time_diff = False for now
    df = df.withColumn("sequence_number", col("sequence_number_ordinal"))

    return df


def state_and_metrics_sparse2dense(
    df, states: List[int], metrics: List[str], multi_steps: Optional[int]
):
    """ Sparse-to-dense preprocessing of Map columns, which are states and metrics.
    For each column of type Map, w/ name X, output two columns.
        Map values are assumed to be scalar. This process is called sparse-to-dense.
        X = {"state_features", "next_state_features", "metrics"}.
        (a) Replace column X with a dense repesentation of the inputted (sparse) map.
            Dense representation is to concatenate map values into a list.
        (b) Create new column X_presence, which is a list of same length as (a) and
            the ith entry is 1 iff the key was present in the original map.
    """
    next_map_udf = make_next_udf(multi_steps, MapType(LongType(), FloatType()))
    df = df.withColumn("next_state_features", next_map_udf("next_state_features"))
    df = df.withColumn("metrics", next_map_udf("metrics"))
    df = make_sparse2dense(df, "state_features", states)
    df = make_sparse2dense(df, "next_state_features", states)
    df = make_sparse2dense(df, "metrics", metrics)
    return df


def discrete_action_preprocessing(
    df, actions: List[str], multi_steps: Optional[int] = None
):
    """
    Inputted actions and possible_actions are strings, which isn't supported
        for PyTorch Tensors. Here, we represent them with LongType.
        (a) action and next_action are strings, so simply return their position
            in the action_space (as given by argument actions).
        (b) possible_actions and possible_next_actions are list of strs, so
            return an existence bitvector of length len(actions), where ith
            index is true iff actions[i] was in the list.

    By-product: output not_terminal from preprocessing actions.
    """

    # turn string actions into indices
    where_udf = make_where_udf(actions)
    df = df.withColumn("action", where_udf("action"))
    next_long_udf = make_next_udf(multi_steps, LongType())
    df = df.withColumn("next_action", where_udf(next_long_udf("next_action")))

    def make_not_terminal_udf(actions: List[str]):
        """ Return true iff next_action is terminal (i.e. idx = len(actions)). """

        def get_not_terminal(next_action):
            return next_action < len(actions)

        return udf(get_not_terminal, BooleanType())

    not_terminal_udf = make_not_terminal_udf(actions)
    df = df.withColumn("not_terminal", not_terminal_udf("next_action"))

    # turn List[str] possible_actions into existence bitvectors
    next_long_arr_udf = make_next_udf(multi_steps, ArrayType(LongType()))
    existence_bitvector_udf = make_existence_bitvector_udf(actions)
    df = df.withColumn(
        "possible_actions_mask", existence_bitvector_udf("possible_actions")
    )
    df = df.withColumn(
        "possible_next_actions_mask",
        existence_bitvector_udf(next_long_arr_udf("possible_next_actions")),
    )
    return df


def parametric_action_preprocessing(
    df,
    actions: List[str],
    multi_steps: Optional[int] = None,
    include_possible_actions: bool = True,
):
    assert (
        not include_possible_actions
    ), "current we don't support include_possible_actions"

    next_map_udf = make_next_udf(multi_steps, MapType(LongType(), FloatType()))
    df = df.withColumn("next_action", next_map_udf("next_action"))

    def make_not_terminal_udf():
        """ Return true iff next_action is an empty map """

        def get_not_terminal(next_action):
            return len(next_action) > 0

        return udf(get_not_terminal, BooleanType())

    not_terminal_udf = make_not_terminal_udf()
    df = df.withColumn("not_terminal", not_terminal_udf("next_action"))

    df = make_sparse2dense(df, "action", actions)
    df = make_sparse2dense(df, "next_action", actions)
    return df


def select_relevant_columns(
    df, discrete_action: bool = True, include_possible_actions: bool = True
):
    """ Select all the relevant columns and perform type conversions. """
    if not discrete_action and include_possible_actions:
        raise NotImplementedError("currently we don't support include_possible_actions")

    select_col_list = [
        col("reward").cast(FloatType()),
        col("state_features").cast(ArrayType(FloatType())),
        col("state_features_presence").cast(ArrayType(BooleanType())),
        col("next_state_features").cast(ArrayType(FloatType())),
        col("next_state_features_presence").cast(ArrayType(BooleanType())),
        col("not_terminal").cast(BooleanType()),
        col("action_probability").cast(FloatType()),
        col("mdp_id").cast(LongType()),
        col("sequence_number").cast(LongType()),
        col("step").cast(LongType()),
        col("time_diff").cast(LongType()),
        col("metrics").cast(ArrayType(FloatType())),
        col("metrics_presence").cast(ArrayType(BooleanType())),
    ]

    if discrete_action:
        select_col_list += [
            col("action").cast(LongType()),
            col("next_action").cast(LongType()),
        ]
    else:
        select_col_list += [
            col("action").cast(ArrayType(FloatType())),
            col("next_action").cast(ArrayType(FloatType())),
            col("action_presence").cast(ArrayType(BooleanType())),
            col("next_action_presence").cast(ArrayType(BooleanType())),
        ]

    if include_possible_actions:
        select_col_list += [
            col("possible_actions_mask").cast(ArrayType(LongType())),
            col("possible_next_actions_mask").cast(ArrayType(LongType())),
        ]

    return df.select(*select_col_list)


def get_distinct_keys(df, col_name, is_col_arr_map=False):
    """ Return list of distinct keys.
        Set is_col_arr_map to be true if column is an array of Maps.
        Otherwise, assume column is a Map.
    """
    if is_col_arr_map:
        df = df.select(explode(col_name).alias(col_name))
    df = df.select(explode(map_keys(col_name)))
    return df.distinct().rdd.flatMap(lambda x: x).collect()


def infer_states_names(df, multi_steps: Optional[int]):
    """ Infer possible state names from states and next state features. """
    state_keys = get_distinct_keys(df, "state_features")
    next_states_is_col_arr_map = not (multi_steps is None)
    next_state_keys = get_distinct_keys(
        df, "next_state_features", is_col_arr_map=next_states_is_col_arr_map
    )
    return sorted(set(state_keys) | set(next_state_keys))


def infer_action_names(df, multi_steps: Optional[int]):
    action_keys = get_distinct_keys(df, "action")
    next_action_is_col_arr_map = not (multi_steps is None)
    next_action_keys = get_distinct_keys(
        df, "next_action", is_col_arr_map=next_action_is_col_arr_map
    )
    return sorted(set(action_keys) | set(next_action_keys))


def infer_metrics_names(df, multi_steps: Optional[int]):
    """ Infer possible metrics names.
    Assume in multi-step case, metrics is an array of maps.
    """
    is_col_arr_map = not (multi_steps is None)
    return sorted(get_distinct_keys(df, "metrics", is_col_arr_map=is_col_arr_map))


def rand_string(length):
    import string
    import random

    """Generate a random string of fixed length """
    r = random.SystemRandom()
    letters = string.ascii_lowercase
    return "".join(r.choice(letters) for _ in range(length))


def upload_as_parquet(df) -> Dataset:
    """ Generate a random parquet. Fails if cannot generate a non-existent name. """

    # get a random tmp name and check if it exists
    sqlCtx = get_spark_session()
    success = False
    for _ in range(MAX_UPLOAD_PARQUET_TRIES):
        suffix = rand_string(length=UPLOAD_PARQUET_TMP_SUFFIX_LEN)
        rand_name = f"tmp_parquet_{suffix}"
        if not sqlCtx.catalog._jcatalog.tableExists(rand_name):
            success = True
            break
    if not success:
        raise Exception(f"Failed to find name after {MAX_UPLOAD_PARQUET_TRIES} tries.")

    # perform the write
    df.write.mode("errorifexists").format("parquet").saveAsTable(rand_name)
    parquet_url = get_table_url(rand_name)
    logger.info(f"Saved parquet to {parquet_url}")
    return OssDataset(parquet_url=parquet_url)


def query_data(
    input_table_spec: TableSpec,
    discrete_action: bool,
    actions: Optional[List[str]] = None,
    include_possible_actions=True,
    custom_reward_expression: Optional[str] = None,
    sample_range: Optional[Tuple[float, float]] = None,
    multi_steps: Optional[int] = None,
    gamma: Optional[float] = None,
) -> Dataset:
    """ Perform reward calculation, hashing mdp + subsampling and
    other preprocessing such as sparse2dense.
    """
    sqlCtx = get_spark_session()
    df = sqlCtx.sql(f"SELECT * FROM {input_table_spec.table}")
    df = set_reward_col_as_reward(
        df,
        custom_reward_expression=custom_reward_expression,
        multi_steps=multi_steps,
        gamma=gamma,
    )
    df = hash_mdp_id_and_subsample(df, sample_range=sample_range)
    df = misc_column_preprocessing(df, multi_steps=multi_steps)
    df = state_and_metrics_sparse2dense(
        df,
        states=infer_states_names(df, multi_steps),
        metrics=infer_metrics_names(df, multi_steps),
        multi_steps=multi_steps,
    )
    if discrete_action:
        assert include_possible_actions
        assert actions is not None, "in discrete case, actions must be given."
        df = discrete_action_preprocessing(df, actions=actions, multi_steps=multi_steps)
    else:
        actions = infer_action_names(df, multi_steps)
        df = parametric_action_preprocessing(
            df,
            actions=actions,
            multi_steps=multi_steps,
            include_possible_actions=include_possible_actions,
        )

    df = select_relevant_columns(
        df,
        discrete_action=discrete_action,
        include_possible_actions=include_possible_actions,
    )
    return upload_as_parquet(df)


def collate_and_preprocess(batch_preprocessor: BatchPreprocessor, use_gpu: bool):
    """ Helper for Petastorm's DataLoader to preprocess.
    TODO(kaiwenw): parallelize preprocessing by using transform of Petastorm reader
    Should pin memory and preprocess in reader and convert to gpu in collate_fn.
    """

    def collate_fn(batch_list: List[Dict]):
        batch = decimal_friendly_collate(batch_list)
        preprocessed_batch = batch_preprocessor(batch)
        if use_gpu:
            preprocessed_batch = preprocessed_batch.cuda()
        return preprocessed_batch

    return collate_fn


class OssDataFetcher(DataFetcher):
    def query_data(self, **kwargs):
        return query_data(**kwargs)

    def query_data_parametric(self, **kwargs):
        return query_data(**kwargs)

    def identify_normalization_parameters(
        self,
        table_spec: TableSpec,
        column_name: str,
        preprocessing_options: PreprocessingOptions,
        seed: Optional[int] = None,
    ) -> Dict[int, NormalizationParameters]:
        return identify_normalization_parameters(
            table_spec, column_name, preprocessing_options, seed
        )

    def get_table_row_count(self, dataset: OssDataset):
        spark = get_spark_session()
        return spark.read.parquet(dataset.parquet_url).count()

    def gather_and_sort_eval_data(
        self,
        trainer: RLTrainer,
        eval_dataset: Dataset,
        batch_preprocessor: BatchPreprocessor,
        use_gpu: bool,
        reader_options: ReaderOptions,
    ) -> EvaluationDataPage:
        """ Sorts, computes logged values and validates the EvaluationDataPage """
        if isinstance(trainer, (SACTrainer, TD3Trainer)):
            raise NotImplementedError("TODO: Implement CPE for continuous algos")
        assert (
            trainer.calc_cpe_in_training
        ), "this function should only be called when this is true."

        # first read the eval_dataset as EvaluationDataPages
        device = "cuda" if use_gpu else "cpu"
        eval_data = None
        with make_batch_reader(
            eval_dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petastorm_reader_pool_type,
        ) as reader:
            for batch in reader:
                assert rlt.isinstance_namedtuple(batch)
                tensor_batch = dict_to_tensor(batch._asdict(), device=device)
                tdp: rlt.PreprocessedTrainingBatch = batch_preprocessor(tensor_batch)
                edp = EvaluationDataPage.create_from_training_batch(tdp, trainer)
                if eval_data is None:
                    eval_data = edp
                else:
                    eval_data = eval_data.append(edp)

        eval_data = eval_data.sort()
        eval_data = eval_data.compute_values(trainer.gamma)
        eval_data.validate()
        return eval_data

    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        batch_preprocessor: Optional[BatchPreprocessor],
        use_gpu: bool,
        reader_options: ReaderOptions,
    ):
        """ get petastorm loader for dataset (with preprocessor) """
        data_reader = make_batch_reader(
            dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petastorm_reader_pool_type,
        )
        # NOTE: must be wrapped by DataLoaderWrapper to call __exit__() on end of epoch
        return DataLoader(
            data_reader,
            batch_size=batch_size,
            collate_fn=collate_and_preprocess(
                batch_preprocessor=batch_preprocessor, use_gpu=use_gpu
            ),
        )

    def get_post_dataloader_preprocessor(
        self, reader_options: ReaderOptions, use_gpu: bool
    ):
        return None
