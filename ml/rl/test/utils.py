#!/usr/bin/env python3

import collections
import csv

from ml.rl.preprocessing.normalization import NormalizationParameters


def default_normalizer(feats, min_value=None, max_value=None):
    normalization = collections.OrderedDict(
        [
            (
                feats[i],
                NormalizationParameters(
                    feature_type="CONTINUOUS",
                    boxcox_lambda=None,
                    boxcox_shift=0,
                    mean=0,
                    stddev=1,
                    possible_values=None,
                    quantiles=None,
                    min_value=min_value,
                    max_value=max_value,
                ),
            )
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
