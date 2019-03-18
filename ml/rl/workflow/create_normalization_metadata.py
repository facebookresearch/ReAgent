#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import logging
import os
import sys
from collections import defaultdict

import pandas as pd
from ml.rl.preprocessing.normalization import (
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_QUANTILE_K2_THRESHOLD,
    get_feature_norm_metadata,
    serialize,
)
from ml.rl.readers.json_dataset_reader import JSONDatasetReader
from ml.rl.workflow.helpers import parse_args


logger = logging.getLogger(__name__)

NORMALIZATION_BATCH_READ_SIZE = 50000


def create_norm_table(params):
    training_data_path = params["training_data_path"]
    logger.info("Generating norm table based on {}".format(training_data_path))

    norm_params = get_norm_params(params["norm_params"])
    dataset = JSONDatasetReader(
        params["training_data_path"], batch_size=NORMALIZATION_BATCH_READ_SIZE
    )

    for col in norm_params["cols_to_norm"]:
        logger.info("Creating normalization metadata for `{}` column".format(col))
        norm_metadata = get_norm_metadata(dataset, norm_params, col)
        path = norm_params["output_dir"] + "{}_norm.json".format(col)
        with open(os.path.expanduser(path), "w") as outfile:
            json.dump(norm_metadata, outfile)
            logger.info("`{}` normalization metadata written to {}".format(col, path))


def get_norm_metadata(dataset, norm_params, norm_col):
    done = False
    batch = dataset.read_batch()
    samples_per_feature, samples = defaultdict(int), defaultdict(list)

    while not done:
        if batch is None or len(batch[norm_col]) == 0:
            logger.info("No more data in training data. Breaking.")
            break

        feature_df = pd.DataFrame.from_dict(batch[norm_col]).apply(pd.Series)
        for feature in feature_df:
            values = feature_df[feature].dropna().values
            samples_per_feature[feature] += len(values)
            samples[feature].extend(values)

        done = check_samples_per_feature(
            samples_per_feature, norm_params["num_samples"]
        )
        logger.info("Samples per feature: {}".format(samples_per_feature))
        if done:
            logger.info("Collected sufficient sample size for all features. Breaking.")

        batch = dataset.read_batch()

    output = {}
    for feature, values in samples.items():
        output[feature] = get_feature_norm_metadata(feature, values, norm_params)
    return serialize(output)


def check_samples_per_feature(samples_per_feature, num_samples):
    for _, v in samples_per_feature.items():
        if v < num_samples:
            return False
    return True


def get_norm_params(norm_params):
    return {
        "num_samples": norm_params.get("num_samples", DEFAULT_NUM_SAMPLES),
        "max_unique_enum_values": norm_params.get(
            "max_unique_enum_values", DEFAULT_MAX_UNIQUE_ENUM
        ),
        "quantile_size": norm_params.get("quantile_size", DEFAULT_MAX_QUANTILE_SIZE),
        "quantile_k2_threshold": norm_params.get(
            "quantile_k2_threshold", DEFAULT_QUANTILE_K2_THRESHOLD
        ),
        "skip_box_cox": norm_params.get("skip_box_cox", False),
        "skip_quantiles": True,  # Skipping quantiles helps performance in OpenAI Gym Cartpole-v0
        "feature_overrides": norm_params.get("feature_overrides", None),
        "cols_to_norm": norm_params["cols_to_norm"],
        "output_dir": norm_params["output_dir"],
    }


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    params = parse_args(sys.argv)
    create_norm_table(params)
