#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import inspect


class DecisionOperator(object):
    def __init__(self):
        self.id = None
        self.op_name = type(self).__name__

    def arguments(self):
        raise NotImplementedError


def DecisionOperation(op):
    def __init__(self, *args, **kwargs):
        self.args = inspect.getcallargs(op, *args, **kwargs)
        DecisionOperator.__init__(self)

    def arguments(self):
        return self.args

    return type(
        op.__name__, (DecisionOperator,), {"__init__": __init__, "arguments": arguments}
    )
