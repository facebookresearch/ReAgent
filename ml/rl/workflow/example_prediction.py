#!/usr/bin/env python3

# Minimal example of how to use Caffe2 net for prediction with a trained
# DQNPredictor on Cartpole-v0.
#
# Usage:
# python ml/rl/workflow/example_prediction.py -p \
#    ml/rl/workflow/sample_datasets/discrete_action/example_predictor.c2


import argparse
import os
import sys

from ml.rl.training.dqn_predictor import DQNPredictor


def main(path):
    path = os.path.expanduser(path)
    predictor = DQNPredictor.load(path, "minidb", int_features=False)
    test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
    print("Q-values: ", predictor.predict(test_float_state_features))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: python <file.py> -p <path_to_c2_predictor>")

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to Caffe2 predictor.")
    args = parser.parse_args(sys.argv[1:])
    main(args.path)
