#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import json
import logging
import os

import torch
from reagent.models.dqn import FullyConnectedDQN
from reagent.training.dqn_trainer import DQNTrainer
from reagent.training.parametric_dqn_trainer import ParametricDQNTrainer


logger = logging.getLogger(__name__)


def minibatch_size_multiplier(use_gpu, use_all_avail_gpus):
    """Increase size of minibatch if using PyTorch DataParallel."""
    if use_gpu and use_all_avail_gpus and torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def parse_args(args):
    if len(args) < 3:
        raise Exception("Usage: python <file.py> -p <parameters_file>")

    parser = argparse.ArgumentParser(description="Read command line parameters.")
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
    parser.add_argument("-ni", "--node_index", help="Node index")
    args = parser.parse_args(args[1:])

    with open(args.parameters, "r") as f:
        params = json.load(f)

    params["node_index"] = args.node_index

    return params


def save_model_to_file(model, path):
    """
    Save network parameters and optimizer parameters to file.

    :param model: one of (DQNTrainer, ParametricDQNTrainer) object.
    """
    path = os.path.expanduser(path)
    try:
        state = model.state_dict()
        torch.save(state, path)
        return
    except NotImplementedError:
        pass

    if isinstance(model, (DQNTrainer, ParametricDQNTrainer)):
        state = {
            "q_network": model.q_network.state_dict(),
            "optimizer": model.q_network_optimizer.state_dict(),
        }
        torch.save(state, path)
    else:
        raise ValueError("Model of type {} not a valid model".format(type(model)))


def update_model_for_warm_start(model, path=None):
    """
    Load network parameters and optimizer parameters into trainer object
    to warm start it.

    :param model: one of (DQNTrainer, ParametricDQNTrainer) object.
    """
    if path is None and getattr(model, "warm_start_model_path", None) is None:
        return model

    if path is None:
        path = model.warm_start_model_path

    path = os.path.expanduser(path)
    state = torch.load(path)
    logger.info("Found model warm start checkpoint at path {}".format(path))

    try:
        model.load_state_dict(state)
        return model
    except NotImplementedError:
        pass

    if isinstance(model, (DQNTrainer, ParametricDQNTrainer)):
        try:
            model.q_network.load_state_dict(state["q_network"])
            model.q_network_target.load_state_dict(state["q_network"])
            if not model.parameters.training.do_not_warm_start_optimizer:
                model.q_network_optimizer.load_state_dict(state["optimizer"])
        except Exception:
            if not isinstance(model.q_network, FullyConnectedDQN):
                raise
            # If it's FullyConnectedDQN, we try to load FullyConnectedNetwork into it
            # by adding "fc." prefix
            state["q_network"] = type(state["q_network"])(
                ("fc.{}".format(k), v) for k, v in state["q_network"].items()
            )
            model.q_network.load_state_dict(state["q_network"])
            model.q_network_target.load_state_dict(state["q_network"])
            if not model.parameters.training.do_not_warm_start_optimizer:
                model.q_network_optimizer.load_state_dict(state["optimizer"])
    else:
        raise ("Model of type {} not a valid model".format(type(model)))

    return model
