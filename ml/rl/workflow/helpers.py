#!/usr/bin/env python3

import argparse
import json
import logging

import torch
from ml.rl.training.ddpg_trainer import DDPGTrainer
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def report_training_status(batch_num, num_batches, epoch_num, num_epochs):
    percent_complete = batch_num / (num_batches * num_epochs) * 100
    logger.info(
        "On batch {} of {} and epoch {} of {} ({}% complete)".format(
            batch_num + 1,
            num_batches,
            epoch_num + 1,
            num_epochs,
            round(percent_complete),
        )
    )


def parse_args(args):

    if len(args) != 3:
        raise Exception("Usage: python <file.py> -p <parameters_file>")

    parser = argparse.ArgumentParser(description="Read command line parameters.")
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
    args = parser.parse_args(args[1:])

    with open(args.parameters, "r") as f:
        params = json.load(f)

    return params


def save_model_to_file(model, path):
    """
    Save network parameters and optimizer parameters to file.

    :param model: one of (DQNTrainer, ParametricDQNTrainer, DDPGTrainer) object.
    """
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


def update_model_for_warm_start(model):
    """
    Load network parameters and optimizer parameters into trainer object
    to warm start it.

    :param model: one of (DQNTrainer, ParametricDQNTrainer, DDPGTrainer) object.
    """

    if model.warm_start_model_path is None:
        return model

    state = torch.load(model.warm_start_model_path)
    logger.info(
        "Found model warm start checkpoint at path {}".format(
            model.warm_start_model_path
        )
    )

    if isinstance(model, (DQNTrainer, ParametricDQNTrainer)):
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

    model.reward_burnin = -1
    return model
