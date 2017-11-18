from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.optimizer import (
    build_sgd, build_ftrl, build_adagrad, build_adam
)

from enum import Enum


class GRAD_OPTIMIZER(Enum):
    SGD = 1
    ADAGRAD = 2
    ADAM = 3
    FTRL = 4


OPTIMIZER_DICT = dict([(op.name, op) for op in GRAD_OPTIMIZER])


def AddParameterUpdateOps(
    model, optimizer_input="SGD", base_learning_rate=0.01, *args, **kwargs
):
    if optimizer_input not in OPTIMIZER_DICT:
        raise Exception(
            "Optimizer {} unknown. Valid choices are {}"
            .format(optimizer_input, ', '.join(OPTIMIZER_DICT.keys()))
        )
    optimizer_rule = OPTIMIZER_DICT[optimizer_input]

    if optimizer_rule == GRAD_OPTIMIZER.SGD:
        build_sgd(
            model,
            base_learning_rate,
            gamma=kwargs['gamma'],
            policy=kwargs['policy'],
            stepsize=1
        )
    elif optimizer_rule == GRAD_OPTIMIZER.ADAGRAD:
        build_adagrad(model, base_learning_rate)
    elif optimizer_rule == GRAD_OPTIMIZER.ADAM:
        build_adam(model, base_learning_rate)
    elif optimizer_rule == GRAD_OPTIMIZER.FTRL:
        build_ftrl(model, base_learning_rate)
    else:
        print(
            "Unrecognized in caffe2 setting, using default SGD", optimizer_rule
        )
        build_sgd(model, base_learning_rate)
