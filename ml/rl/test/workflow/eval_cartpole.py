#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import logging
import sys

from ml.rl.test.gym.open_ai_gym_environment import OpenAIGymEnvironment
from ml.rl.training.dqn_predictor import DQNPredictor


logger = logging.getLogger(__name__)

ENV = "CartPole-v0"
AVG_OVER_NUM_EPS = 100


def main(model_path):
    predictor = DQNPredictor.load(model_path, "minidb")

    env = OpenAIGymEnvironment(gymenv=ENV)

    avg_rewards, avg_discounted_rewards = env.run_ep_n_times(
        AVG_OVER_NUM_EPS, predictor, test=True
    )

    logger.info(
        "Achieved an average reward score of {} over {} evaluations.".format(
            avg_rewards, AVG_OVER_NUM_EPS
        )
    )


def parse_args(args):
    if len(args) != 3:
        raise Exception("Usage: python <file.py> -m <parameters_file>")

    parser = argparse.ArgumentParser(description="Read command line parameters.")
    parser.add_argument("-m", "--model", help="Path to Caffe2 model.")
    args = parser.parse_args(args[1:])
    return args.model


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    model_path = parse_args(sys.argv)
    main(model_path)
