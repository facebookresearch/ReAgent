#!/usr/bin/env python3

import collections
import csv

from ml.rl.preprocessing.normalization import NormalizationParameters


def default_normalizer(feats, min_value=None, max_value=None):
    normalization_types = [
        NormalizationParameters(
            feature_type="BINARY",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="PROBABILITY",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="CONTINUOUS",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="BOXCOX",
            boxcox_lambda=1,
            boxcox_shift=1,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="QUANTILE",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=[0, 1],
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="ENUM",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=[0, 1],
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
    ]
    normalization = collections.OrderedDict(
        [
            (feats[i], normalization_types[i % len(normalization_types)])
            for i in range(len(feats))
        ]
    )
    return normalization


def write_lists_to_csv(path, *args):
    rows = zip(*args)
    with open(path, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
