#!/usr/bin/env python3


from typing import List

from enum import Enum

from caffe2.python.model_helper import ModelHelper
from caffe2.python.optimizer import build_sgd, build_ftrl, build_adagrad, build_adam

from ml.rl.thrift.core.ttypes import TrainingParameters
from ml.rl.training.dnn import DNN


class GRAD_OPTIMIZER(Enum):
    SGD = 1
    ADAGRAD = 2
    ADAM = 3
    FTRL = 4


OPTIMIZER_DICT = {op.name: op for op in GRAD_OPTIMIZER}


class MLTrainer(DNN):
    """ This is meant to be a generic neural net trainer.  It uses minibatch and
    ADAM for momentum/smoothing.
    """

    def __init__(self, name: str, parameters: TrainingParameters) -> None:
        """

        :param name: A unique name for this trainer used to create the data on the
            caffe2 workspace
        :param parameters: The set of training parameters
        """
        self.optimizer = parameters.optimizer
        self.learning_rate = parameters.learning_rate

        self.lr_decay = parameters.lr_decay
        self.lr_policy = parameters.lr_policy

        DNN.__init__(self, name, parameters)

    def generateLossOps(
        self, model: ModelHelper, output_blob: str, label_blob: str
    ) -> str:
        """
        Adds loss operators to net. The loss function is computed by a squared L2
        distance, and then averaged over all items in the minibatch.

        :param model: ModelHelper object to add loss operators to.
        :param model_id: String identifier.
        :param output_blob: Blob containing output of net.
        :param label_blob: Blob containing labels.
        :param loss_blob: Blob in which to store loss.
        """
        dist = model.SquaredL2Distance(
            [label_blob, output_blob], model.net.NextBlob("dist")
        )
        loss = model.net.NextBlob("loss")
        model.AveragedLoss(dist, loss)
        return loss

    def addParameterUpdateOps(self, model):
        if self.optimizer not in OPTIMIZER_DICT:
            raise Exception(
                "Optimizer {} unknown. Valid choices are {}".format(
                    self.optimizer, ", ".join(OPTIMIZER_DICT.keys())
                )
            )
        optimizer_rule = OPTIMIZER_DICT[self.optimizer]

        if optimizer_rule == GRAD_OPTIMIZER.SGD:
            build_sgd(
                model,
                self.learning_rate,
                gamma=self.lr_decay,
                policy=self.lr_policy,
                stepsize=1,
            )
        elif optimizer_rule == GRAD_OPTIMIZER.ADAGRAD:
            build_adagrad(model, self.learning_rate)
        elif optimizer_rule == GRAD_OPTIMIZER.ADAM:
            build_adam(model, self.learning_rate)
        elif optimizer_rule == GRAD_OPTIMIZER.FTRL:
            build_ftrl(model, self.learning_rate)
        else:
            print("Unrecognized in caffe2 setting, using default SGD", optimizer_rule)
            build_sgd(model, self.learning_rate)

    def build_predictor(self, model, input_blob, output_blob) -> List[str]:
        self.make_forward_pass_ops(model, input_blob, output_blob, is_test=True)
        return self.weights + self.biases
