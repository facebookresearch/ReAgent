#!/usr/bin/env python3

import logging
import sys

from ml.rl.workflow.helpers import parse_args


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def train_network(params):
    logger.info("Running Parametric DQN workflow with params:")
    logger.info(params)
    pass


if __name__ == "__main__":
    params = parse_args(sys.argv)
    train_network(params)
