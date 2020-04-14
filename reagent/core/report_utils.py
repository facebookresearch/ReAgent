#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from math import ceil
from typing import Dict, List

import numpy as np


logger = logging.getLogger(__name__)


def get_mean_of_recent_values(
    values: Dict[str, List[float]], min_window_size=10
) -> Dict[str, float]:
    ret = {}
    for key, vals in values.items():
        window_size = max(min_window_size, int(ceil(0.1 * len(vals))))
        ret[key] = np.mean(vals[-window_size:])
    return ret


def calculate_recent_window_average(arr, window_size, num_entries):
    if len(arr) > 0:
        begin = max(0, len(arr) - window_size)
        return np.mean(np.array(arr[begin:]), axis=0)
    else:
        logger.error("Not enough samples for evaluation.")
        if num_entries == 1:
            return float("nan")
        else:
            return [float("nan")] * num_entries
