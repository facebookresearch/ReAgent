#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import gym
import torch
from reagent.gym.preprocessors import make_trainer_preprocessor_online
from reagent.gym.types import Trajectory
from reagent.training.trainer import Trainer


def train_post_episode(env: gym.Env, trainer: Trainer, use_gpu: bool):
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    trainer_preprocessor = make_trainer_preprocessor_online(trainer, device, env)

    def post_episode(trajectory: Trajectory):
        training_batch = trainer_preprocessor(trajectory)
        trainer.train(training_batch)

    return post_episode
