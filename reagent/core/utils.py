#!/usr/bin/env python3

from typing import Tuple, NamedTuple

from reagent.workflow.types import (
    TableSpec,
)


class lazy_property(object):
    """
    More or less copy-pasta: http://stackoverflow.com/a/6849299
    Meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self._fget = fget
        self.__doc__ = fget.__doc__
        self.__name__ = fget.__name__

    def __get__(self, obj, obj_cls_type):
        if obj is None:
            return None
        value = self._fget(obj)
        setattr(obj, self.__name__, value)
        return value


class TrainEvalSampleRanges(NamedTuple):
    train_sample_range: Tuple[float, float]
    eval_sample_range: Tuple[float, float]


def get_sample_range(
    input_table_spec: TableSpec, calc_cpe_in_training: bool
) -> TrainEvalSampleRanges:
    table_sample = input_table_spec.table_sample
    eval_dataset = input_table_spec.eval_dataset
    eval_table_sample = input_table_spec.eval_table_sample

    if not calc_cpe_in_training:
        # use all data if table sample = None
        if table_sample is None:
            train_sample_range = (0.0, 100.0)
        else:
            train_sample_range = (0.0, table_sample)
        return TrainEvalSampleRanges(
            train_sample_range=train_sample_range,
            # eval samples will not be used
            eval_sample_range=(0.0, 0.0),
        )

    error_msg = (
        "calc_cpe_in_training is set to True. "
        "Please specify eval_table in input_table_spec. Alternatively"
        "you can split eval dataset from input_table_spec.dataset, but"
        f"please specify table_sample(current={table_sample}) and "
        f"eval_table_sample(current={eval_table_sample}) such that "
        "eval_table_sample + table_sample <= 100. "
        "In order to reliably calculate CPE, eval_table_sample "
        "should not be too small."
    )
    eval_table_sample = 100.0 if eval_table_sample is None else eval_table_sample
    table_sample = 100.0 if table_sample is None else table_sample

    assert table_sample <= 100.0 + 1e-3 and eval_table_sample <= 100.0 + 1e-3, error_msg
    assert eval_dataset is not None or (eval_table_sample + table_sample) <= (
        100.0 + 1e-3
    ), error_msg

    return TrainEvalSampleRanges(
        train_sample_range=(0.0, table_sample),
        eval_sample_range=(100.0 - eval_table_sample, 100.0),
    )
