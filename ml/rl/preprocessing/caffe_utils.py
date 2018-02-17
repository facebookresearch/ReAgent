#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import itertools
from typing import List, Dict, Tuple

from caffe2.python import workspace


def dict_list_to_blobs(
    d: List[Dict[int, float]],
    blob_prefix: str,
) -> Tuple[str, str, str]:
    lengths_blob = blob_prefix + "_lengths"
    keys_blob = blob_prefix + "_keys"
    values_blob = blob_prefix + "_values"

    workspace.FeedBlob(
        lengths_blob, np.array([len(x) for x in d], dtype=np.int32)
    )

    key_list_2d = [list(x.keys()) for x in d]
    workspace.FeedBlob(
        keys_blob,
        np.array(list(itertools.chain(*key_list_2d)), dtype=np.int32)
    )

    value_list_2d = [list(x.values()) for x in d]
    workspace.FeedBlob(
        values_blob,
        np.array(list(itertools.chain(*value_list_2d)), dtype=np.float32)
    )

    return lengths_blob, keys_blob, values_blob
