#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import Dict, List, Optional

import reagent.core.types as rlt

from pyspark.sql.functions import col, collect_list, explode
from reagent.data.spark_utils import get_spark_session
from reagent.preprocessing.normalization import (
    get_feature_norm_metadata,
    NormalizationParameters,
)

from .types import PreprocessingOptions, TableSpec


def normalization_helper(
    max_unique_enum_values: int,
    quantile_size: int,
    quantile_k2_threshold: float,
    skip_box_cox: bool = False,
    skip_quantiles: bool = False,
    feature_overrides: Optional[Dict[int, str]] = None,
    allowedlist_features: Optional[List[int]] = None,
    assert_allowedlist_feature_coverage: bool = True,
):
    """Construct a preprocessing closure to obtain normalization parameters
    from rows of feature_name and a sample of feature_values.
    """

    norm_params = {
        "max_unique_enum_values": max_unique_enum_values,
        "quantile_size": quantile_size,
        "quantile_k2_threshold": quantile_k2_threshold,
        "skip_box_cox": skip_box_cox,
        "skip_quantiles": skip_quantiles,
        "feature_overrides": feature_overrides,
    }
    # pyre-fixme[9]: allowedlist_features has type `Optional[List[int]]`; used as
    #  `Set[int]`.
    # pyre-fixme[9]: allowedlist_features has type `Optional[List[int]]`; used as
    #  `Set[int]`.
    allowedlist_features = set(allowedlist_features or [])

    def validate_allowedlist_features(
        params: Dict[int, NormalizationParameters],
    ) -> None:
        if not allowedlist_features:
            return
        allowedlist_feature_set = {int(fid) for fid in allowedlist_features}
        available_features = set(params.keys())
        assert allowedlist_feature_set == available_features, (
            "Could not identify preprocessing type for these features: {}; "
            "extra features: {}".format(
                allowedlist_feature_set - available_features,
                available_features - allowedlist_feature_set,
            )
        )

    def process(rows: List) -> Dict[int, NormalizationParameters]:
        params = {}
        for row in rows:
            assert "feature_name" in row
            assert "feature_values" in row
            norm_metdata = get_feature_norm_metadata(
                row["feature_name"], row["feature_values"], norm_params
            )
            if norm_metdata is not None and (
                not allowedlist_features or row["feature_name"] in allowedlist_features
            ):
                params[row["feature_name"]] = norm_metdata

        if assert_allowedlist_feature_coverage:
            validate_allowedlist_features(params)
        return params

    return process


def identify_normalization_parameters(
    table_spec: TableSpec,
    column_name: str,
    preprocessing_options: PreprocessingOptions,
    seed: Optional[int] = None,
) -> Dict[int, NormalizationParameters]:
    """Get normalization parameters"""
    sqlCtx = get_spark_session()
    df = sqlCtx.sql(f"SELECT * FROM {table_spec.table_name}")
    df = create_normalization_spec_spark(
        df, column_name, preprocessing_options.num_samples, seed
    )
    rows = df.collect()

    normalization_processor = normalization_helper(
        max_unique_enum_values=preprocessing_options.max_unique_enum_values,
        quantile_size=preprocessing_options.quantile_size,
        quantile_k2_threshold=preprocessing_options.quantile_k2_threshold,
        skip_box_cox=preprocessing_options.skip_box_cox,
        skip_quantiles=preprocessing_options.skip_quantiles,
        feature_overrides=preprocessing_options.feature_overrides,
        allowedlist_features=preprocessing_options.allowedlist_features,
        assert_allowedlist_feature_coverage=preprocessing_options.assert_allowedlist_feature_coverage,
    )
    return normalization_processor(rows)


def create_normalization_spec_spark(
    df, column, num_samples: int, seed: Optional[int] = None
):
    """Returns approximately num_samples random rows from column of df."""

    # assumes column has a type of map
    df = df.select(
        explode(col(column).alias("features")).alias("feature_name", "feature_value")
    )

    # calculate fractions
    counts_df = df.groupBy("feature_name").count()
    frac = {}
    for row in counts_df.collect():
        assert num_samples <= row["count"]
        frac[row["feature_name"]] = num_samples / row["count"]

    # TODO(T64843081): change to reservoir sampling, currently it approximates
    # perform sampling and collect them
    df = df.sampleBy("feature_name", fractions=frac, seed=seed)
    df = df.groupBy("feature_name").agg(
        collect_list("feature_value").alias("feature_values")
    )
    return df


# TODO: for OSS
def identify_sparse_normalization_parameters(
    feature_config: rlt.ModelFeatureConfig,
    table_spec: TableSpec,
    id_list_column: str,
    id_score_list_column: str,
    preprocessing_options: PreprocessingOptions,
):
    return {}
