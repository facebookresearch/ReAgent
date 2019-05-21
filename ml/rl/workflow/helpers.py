#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import json
import logging
import os
import time

import torch
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.training.ddpg_trainer import DDPGTrainer
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


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

    :param model: one of (DQNTrainer, ParametricDQNTrainer, DDPGTrainer) object.
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
    elif isinstance(model, DDPGTrainer):
        state = {
            "actor": model.actor.state_dict(),
            "actor_optimizer": model.actor_optimizer.state_dict(),
            "critic": model.critic.state_dict(),
            "critic_optimizer": model.critic_optimizer.state_dict(),
        }
        torch.save(state, path)
    else:
        raise ("Model of type {} not a valid model".format(type(model)))


def update_model_for_warm_start(model, path=None):
    """
    Load network parameters and optimizer parameters into trainer object
    to warm start it.

    :param model: one of (DQNTrainer, ParametricDQNTrainer, DDPGTrainer) object.
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
            model.q_network_optimizer.load_state_dict(state["optimizer"])
    elif isinstance(model, DDPGTrainer):
        model.actor.load_state_dict(state["actor"])
        model.actor_target.load_state_dict(state["actor"])
        model.actor_optimizer.load_state_dict(state["actor_optimizer"])
        model.critic.load_state_dict(state["critic"])
        model.critic_target.load_state_dict(state["critic"])
        model.critic_optimizer.load_state_dict(state["critic_optimizer"])
    else:
        raise ("Model of type {} not a valid model".format(type(model)))

    return model


def export_trainer_and_predictor(trainer, output_path, exporter=None):
    """Writes PyTorch trainer and Caffe2 Predictor to file and returns predictor

    returns: Predictor object
    """
    export_time = round(time.time())

    if output_path is None:
        # Don't write models to file, just return predictor
        caffe2_predictor = exporter.export() if exporter else trainer.predictor()
    else:
        # Write models to file and return predictor
        output_path = os.path.expanduser(output_path)
        pytorch_output_path = os.path.join(
            output_path, "trainer_{}.pt".format(export_time)
        )
        caffe2_output_path = os.path.join(
            output_path, "predictor_{}.c2".format(export_time)
        )
        logger.info("Saving PyTorch trainer to {}".format(pytorch_output_path))
        save_model_to_file(trainer, pytorch_output_path)
        logger.info("Saving Caffe2 predictor to {}".format(caffe2_output_path))
        caffe2_predictor = exporter.export() if exporter else trainer.predictor()
        caffe2_predictor.save(caffe2_output_path, "minidb")
    return caffe2_predictor
