#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Get default preprocessors for training time. """

import inspect
import logging

import gym
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# This is here to make typechecker happpy, sigh
MAKER_MAP = {}


def make_replay_buffer_trainer_preprocessor(
    trainer: Trainer, device: torch.device, env: gym.Env
):
    sig = inspect.signature(trainer.train)
    logger.info(f"Deriving trainer_preprocessor from {sig.parameters}")
    # Assuming training_batch is in the first position (excluding self)
    assert (
        list(sig.parameters.keys())[0] == "training_batch"
    ), f"{sig.parameters} doesn't have training batch in first position."
    training_batch_type = sig.parameters["training_batch"].annotation
    assert training_batch_type != inspect.Parameter.empty
    try:
        maker = MAKER_MAP[training_batch_type].create_for_env(env)
    except KeyError:
        logger.error(f"Unknown type: {training_batch_type}")
        raise

    def trainer_preprocessor(batch):
        retval = maker(batch)
        return retval.to(device)

    return trainer_preprocessor


class DiscreteDqnInputMaker:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        return cls(num_actions=action_space.n)

    def __call__(self, batch):
        not_terminal = 1.0 - batch.terminal.float()
        try:
            action = F.one_hot(batch.action, self.num_actions).squeeze(1).float()
            next_action = (
                F.one_hot(batch.next_action, self.num_actions).squeeze(1).float()
            )
        except Exception:
            logger.info(f"action {batch.action}")
            logger.info(f"next_action {batch.next_action}")
            logger.info(self.num_actions)
            raise
        return rlt.DiscreteDqnInput(
            state=rlt.FeatureData(float_features=batch.state),
            action=action,
            next_state=rlt.FeatureData(float_features=batch.next_state),
            next_action=next_action,
            possible_actions_mask=torch.ones_like(action).float(),
            possible_next_actions_mask=torch.ones_like(next_action).float(),
            reward=batch.reward,
            not_terminal=not_terminal,
            step=None,
            time_diff=None,
            extras=rlt.ExtraData(
                mdp_id=None,
                sequence_number=None,
                action_probability=batch.log_prob.exp(),
                max_num_actions=None,
                metrics=None,
            ),
        )


class PolicyNetworkInputMaker:
    @classmethod
    def create_for_env(cls, env: gym.Env):
        return cls()

    def __call__(self, batch):
        not_terminal = 1.0 - batch.terminal.float()
        # TODO: We need to normalized the action in here
        return rlt.PolicyNetworkInput(
            state=rlt.FeatureData(float_features=batch.state),
            action=rlt.FeatureData(float_features=batch.action),
            next_state=rlt.FeatureData(float_features=batch.next_state),
            next_action=rlt.FeatureData(float_features=batch.next_action),
            reward=batch.reward,
            not_terminal=not_terminal,
            step=None,
            time_diff=None,
            extras=rlt.ExtraData(
                mdp_id=None,
                sequence_number=None,
                action_probability=batch.log_prob.exp(),
                max_num_actions=None,
                metrics=None,
            ),
        )


class PreprocessedMemoryNetworkInputMaker:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        return cls(action_space.n)

    def __call__(self, batch):
        # RB's state is (batch_size, state_dim, stack_size) whereas
        # we want (stack_size, batch_size, state_dim)
        # for scalar fields like reward and terminal,
        # RB returns (batch_size, stack_size), where as
        # we want (stack_size, batch_size)
        # Also convert action to float

        if len(batch.state.shape) == 2:
            # this is stack_size = 1 (i.e. we squeezed in RB)
            state = batch.state.unsqueeze(2)
            next_state = batch.next_state.unsqueeze(2)
        else:
            # shapes should be
            state = batch.state
            next_state = batch.next_state
        # Now shapes should be (batch_size, state_dim, stack_size)
        # Turn shapes into (stack_size, batch_size, feature_dim) where
        # feature \in {state, action}; also, make action a float
        permutation = [2, 0, 1]
        not_terminal = 1.0 - batch.terminal.transpose(0, 1).float()
        batch_action = batch.action
        if batch_action.ndim == 3:
            batch_action = batch_action.squeeze(1)
        action = F.one_hot(batch_action, self.num_actions).transpose(1, 2).float()
        return rlt.PreprocessedMemoryNetworkInput(
            state=rlt.FeatureData(state.permute(permutation)),
            next_state=rlt.FeatureData(next_state.permute(permutation)),
            action=action.permute(permutation).float(),
            reward=batch.reward.transpose(0, 1),
            not_terminal=not_terminal,
            step=None,
            time_diff=None,
        )


MAKER_MAP = {
    rlt.DiscreteDqnInput: DiscreteDqnInputMaker,
    rlt.PolicyNetworkInput: PolicyNetworkInputMaker,
    rlt.PreprocessedMemoryNetworkInput: PreprocessedMemoryNetworkInputMaker,
}
