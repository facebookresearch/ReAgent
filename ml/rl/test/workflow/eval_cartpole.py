#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import glob
import logging
import sys

import torch
from ml.rl.prediction.dqn_torch_predictor import DiscreteDqnTorchPredictor
from ml.rl.test.gym.open_ai_gym_environment import OpenAIGymEnvironment


logger = logging.getLogger(__name__)

ENV = "CartPole-v0"
AVG_OVER_NUM_EPS = 100


def main(model_path, temperature):
    model_path = glob.glob(model_path)[0]
    predictor = DiscreteDqnTorchPredictor(torch.jit.load(model_path))
    predictor.softmax_temperature = temperature

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
    parser = argparse.ArgumentParser(description="Read command line parameters.")
    parser.add_argument(
        "-m", "--model", help="Path to TorchScript model.", required=True
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        help="Temperature of softmax",
        required=True,
    )
    return parser.parse_args(args[1:])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args(sys.argv)
    main(args.model, args.softmax_temperature)
