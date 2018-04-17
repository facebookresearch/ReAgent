#!/usr/bin/env python3

import collections

from ml.rl.preprocessing.normalization import NormalizationParameters


def default_normalizer(feats):
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
                )
            ) for i in range(len(feats))
        ]
    )
    return normalization
