#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from functools import partial
from typing import Any, Dict, List

import cloudpickle


def deserialize_and_run(
    fn,
    serialized_args: List[bytes],
    serialized_kwargs: Dict[str, bytes],
    *args,
    **kwargs
):
    d_args: List[Any] = []
    for a in serialized_args:
        d_args.append(cloudpickle.loads(a))
    d_kwargs: Dict[str, Any] = {}
    for k, v in serialized_kwargs.items():
        d_kwargs[k] = cloudpickle.loads(v)
    return cloudpickle.dumps(fn(*d_args, *args, **d_kwargs, **kwargs))


def wrap_function_arguments(fn, *args, **kwargs):
    d_args: List[Any] = []
    for a in args:
        d_args.append(cloudpickle.dumps(a))
    d_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        d_kwargs[k] = cloudpickle.dumps(v)

    return partial(deserialize_and_run, fn, d_args, d_kwargs)


def unwrap_function_outputs(outputs: List[bytes]):
    retval: List[Any] = []
    for o in outputs:
        retval.append(cloudpickle.loads(o))
    return retval
