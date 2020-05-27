#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Union

import numpy as np


def rescale_actions(
    actions: np.ndarray,
    new_min: Union[np.ndarray, float],
    new_max: Union[np.ndarray, float],
    prev_min: Union[np.ndarray, float],
    prev_max: Union[np.ndarray, float],
) -> np.ndarray:
    """ Scale from [prev_min, prev_max] to [new_min, new_max] """
    # pyre-fixme[6]: Expected `float` for 1st param but got `ndarray`.
    assert np.all(prev_min <= actions) and np.all(
        actions <= prev_max
    ), f"{actions} has values outside of [{prev_min}, {prev_max}]."
    assert np.all(
        new_min
        # pyre-fixme[6]: Expected `float` for 1st param but got `Union[float,
        #  np.ndarray]`.
        <= new_max
    ), f"{new_min} is (has coordinate) greater than {new_max}."
    # pyre-fixme[6]: Expected `float` for 1st param but got `Union[float, np.ndarray]`.
    prev_range = prev_max - prev_min
    # pyre-fixme[6]: Expected `float` for 1st param but got `Union[float, np.ndarray]`.
    new_range = new_max - new_min
    return ((actions - prev_min) / prev_range) * new_range + new_min
