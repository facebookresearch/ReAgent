#!/usr/bin/env python3


from typing import List

from ml.rl.thrift.core.ttypes import CNNParameters
from ml.rl.training.conv.cnn import CNN


class ConvMLTrainer(CNN):
    def __init__(self, name: str, cnn_parameters: CNNParameters) -> None:
        CNN.__init__(self, name, cnn_parameters)

    def build_predictor(self, model, input_blob, output_blob) -> List[str]:
        self.make_conv_pass_ops(model, input_blob, output_blob)
        return self.weights + self.biases
