from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

from ml.rl.preprocessing.normalization import \
    NormalizationParameters


def default_normalizer(feats):
    # only for one hot
    normalization = collections.OrderedDict(
        [
            (
                feats[i], NormalizationParameters(
                    feature_type="CONTINUOUS",
                    boxcox_lambda=None,
                    boxcox_shift=0,
                    mean=0,
                    stddev=1
                )
            ) for i in range(len(feats))
        ]
    )
    return normalization
