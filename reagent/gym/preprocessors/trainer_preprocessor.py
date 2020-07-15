#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Get default preprocessors for training time. """

import inspect
import logging
from typing import Optional

import gym
import numpy as np
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.parameters import CONTINUOUS_TRAINING_ACTION_RANGE
from reagent.training.trainer import Trainer
from reagent.training.utils import rescale_actions


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


def one_hot_actions(
    num_actions: int,
    action: torch.Tensor,
    next_action: torch.Tensor,
    terminal: torch.Tensor,
):
    """
    One-hot encode actions and non-terminal next actions.
    Input shape is (batch_size, 1). Output shape is (batch_size, num_actions)
    """
    assert (
        len(action.shape) == 2
        and action.shape[1] == 1
        and next_action.shape == action.shape
    ), (
        f"Must be action with stack_size = 1, but "
        f"got shapes {action.shape}, {next_action.shape}"
    )
    action = F.one_hot(action, num_actions).squeeze(1).float()
    # next action is garbage for terminal transitions (so just zero them)
    next_action_res = torch.zeros_like(action)
    non_terminal_indices = (terminal == 0).squeeze(1)
    next_action_res[non_terminal_indices] = (
        F.one_hot(next_action[non_terminal_indices], num_actions).squeeze(1).float()
    )
    return action, next_action_res


class DiscreteDqnInputMaker:
    def __init__(self, num_actions: int, trainer_preprocessor=None):
        self.num_actions = num_actions
        self.trainer_preprocessor = trainer_preprocessor

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        try:
            return cls(
                num_actions=action_space.n,
                # pyre-fixme[16]: `Env` has no attribute `trainer_preprocessor`.
                trainer_preprocessor=env.trainer_preprocessor,
            )
        except AttributeError:
            return cls(num_actions=action_space.n)

    def __call__(self, batch):
        not_terminal = 1.0 - batch.terminal.float()
        action, next_action = one_hot_actions(
            self.num_actions, batch.action, batch.next_action, batch.terminal
        )
        if self.trainer_preprocessor is not None:
            state = self.trainer_preprocessor(batch.state)
            next_state = self.trainer_preprocessor(batch.next_state)
        else:
            state = rlt.FeatureData(float_features=batch.state)
            next_state = rlt.FeatureData(float_features=batch.next_state)

        return rlt.DiscreteDqnInput(
            state=state,
            action=action,
            next_state=next_state,
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
    def __init__(self, action_low: np.ndarray, action_high: np.ndarray):
        self.action_low = torch.tensor(action_low)
        self.action_high = torch.tensor(action_high)
        (train_low, train_high) = CONTINUOUS_TRAINING_ACTION_RANGE
        self.train_low = torch.tensor(train_low)
        self.train_high = torch.tensor(train_high)

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box)
        return cls(action_space.low, action_space.high)

    def __call__(self, batch):
        not_terminal = 1.0 - batch.terminal.float()
        # normalize actions
        action = rescale_actions(
            batch.action,
            new_min=self.train_low,
            new_max=self.train_high,
            prev_min=self.action_low,
            prev_max=self.action_high,
        )
        # only normalize non-terminal
        non_terminal_indices = (batch.terminal == 0).squeeze(1)
        next_action = torch.zeros_like(action)
        next_action[non_terminal_indices] = rescale_actions(
            batch.next_action[non_terminal_indices],
            new_min=self.train_low,
            new_max=self.train_high,
            prev_min=self.action_low,
            prev_max=self.action_high,
        )
        return rlt.PolicyNetworkInput(
            state=rlt.FeatureData(float_features=batch.state),
            action=rlt.FeatureData(float_features=action),
            next_state=rlt.FeatureData(float_features=batch.next_state),
            next_action=rlt.FeatureData(float_features=next_action),
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


class SlateQInputMaker:
    def __init__(self):
        self.metric = "watch_time"

    @classmethod
    def create_for_env(cls, env: gym.Env):
        return cls()

    def __call__(self, batch):
        n = batch.state.shape[0]
        item_mask = torch.ones(batch.doc.shape[:2])
        next_item_mask = torch.ones(batch.doc.shape[:2])
        # TODO: abs value to make probability?
        item_probability = batch.augmentation_value  # .unsqueeze(2)
        next_item_probability = batch.next_augmentation_value  # .unsqueeze(2)

        # concat null action
        null_action = torch.tensor([batch.action.shape[1]] * n, dtype=torch.int64).view(
            n, 1
        )
        action = torch.cat([batch.action, null_action], dim=1)
        next_action = torch.cat([batch.next_action, null_action], dim=1)

        # concat null reward to position wise reward
        position_reward = getattr(batch, f"response_{self.metric}")
        null_reward = torch.zeros((n, 1))
        position_reward = torch.cat([position_reward, null_reward], dim=1)

        # concat null mask when nothing clicked
        reward_mask = batch.response_click
        null_mask = (reward_mask.sum(dim=1) == 0).view(n, 1)
        reward_mask = torch.cat([reward_mask.to(torch.bool), null_mask], dim=1)
        dict_batch = {
            "state_features": batch.state,
            "next_state_features": batch.next_state,
            "candidate_features": batch.doc,
            "next_candidate_features": batch.next_doc,
            "item_mask": item_mask,
            "next_item_mask": next_item_mask,
            "item_probability": item_probability,
            "next_item_probability": next_item_probability,
            "action": action,
            "next_action": next_action,
            "position_reward": position_reward,
            "reward_mask": reward_mask,
            "time_diff": None,
            "not_terminal": ~batch.terminal,
        }
        return rlt.SlateQInput.from_dict(dict_batch)


class MemoryNetworkInputMaker:
    def __init__(self, num_actions: Optional[int] = None):
        self.num_actions = num_actions

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            return cls(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            return cls()
        else:
            raise NotImplementedError()

    def __call__(self, batch):
        action = batch.action
        if self.num_actions is not None:
            assert len(action.shape) == 2, f"{action.shape}"
            # one hot makes shape (batch_size, stack_size, feature_dim)
            action = F.one_hot(batch.action, self.num_actions).float()
            # make shape to (batch_size, feature_dim, stack_size)
            action = action.transpose(1, 2)

        # For (1-dimensional) vector fields, RB returns (batch_size, state_dim)
        # or (batch_size, state_dim, stack_size).
        # We want these to all be (stack_size, batch_size, state_dim), so
        # unsqueeze the former case; Note this only happens for stack_size = 1.
        # Then, permute.
        permutation = [2, 0, 1]
        vector_fields = {
            "state": batch.state,
            "action": action,
            "next_state": batch.next_state,
        }
        for name, tensor in vector_fields.items():
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(2)
            assert len(tensor.shape) == 3, f"{name} has shape {tensor.shape}"
            vector_fields[name] = tensor.permute(permutation)

        # For scalar fields, RB returns (batch_size), or (batch_size, stack_size)
        # Do same as above, except transpose instead.
        scalar_fields = {
            "reward": batch.reward,
            "not_terminal": 1.0 - batch.terminal.float(),
        }
        for name, tensor in scalar_fields.items():
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(1)
            assert len(tensor.shape) == 2, f"{name} has shape {tensor.shape}"
            scalar_fields[name] = tensor.transpose(0, 1)

        # stack_size > 1, so let's pad not_terminal with 1's, since
        # previous states couldn't have been terminal..
        if scalar_fields["reward"].shape[0] > 1:
            batch_size = scalar_fields["reward"].shape[1]
            assert scalar_fields["not_terminal"].shape == (
                1,
                batch_size,
            ), f"{scalar_fields['not_terminal'].shape}"
            stacked_not_terminal = torch.ones_like(scalar_fields["reward"])
            stacked_not_terminal[-1] = scalar_fields["not_terminal"]
            scalar_fields["not_terminal"] = stacked_not_terminal

        return rlt.MemoryNetworkInput(
            state=rlt.FeatureData(float_features=vector_fields["state"]),
            next_state=rlt.FeatureData(float_features=vector_fields["next_state"]),
            action=vector_fields["action"],
            reward=scalar_fields["reward"],
            not_terminal=scalar_fields["not_terminal"],
            step=None,
            time_diff=None,
        )


def get_possible_actions_for_gym(batch_size: int, num_actions: int) -> rlt.FeatureData:
    """
    tiled_actions should be (batch_size * num_actions, num_actions)
    forall i in [batch_size],
    tiled_actions[i*num_actions:(i+1)*num_actions] should be I[num_actions]
    where I[n] is the n-dimensional identity matrix.
    NOTE: this is only the case for when we convert discrete action to
    parametric action via one-hot encoding.
    """
    possible_actions = torch.eye(num_actions).repeat(repeats=(batch_size, 1))
    return rlt.FeatureData(float_features=possible_actions)


class ParametricDqnInputMaker:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        return cls(action_space.n)

    def __call__(self, batch):
        not_terminal = 1.0 - batch.terminal.float()
        assert (
            len(batch.state.shape) == 2
        ), f"{batch.state.shape} is not (batch_size, state_dim)."
        batch_size, _ = batch.state.shape
        action, next_action = one_hot_actions(
            self.num_actions, batch.action, batch.next_action, batch.terminal
        )
        possible_actions = get_possible_actions_for_gym(batch_size, self.num_actions)
        possible_next_actions = possible_actions.clone()
        possible_actions_mask = torch.ones((batch_size, self.num_actions))
        possible_next_actions_mask = possible_actions_mask.clone()
        return rlt.ParametricDqnInput(
            state=rlt.FeatureData(float_features=batch.state),
            action=rlt.FeatureData(float_features=action),
            next_state=rlt.FeatureData(float_features=batch.next_state),
            next_action=rlt.FeatureData(float_features=next_action),
            possible_actions=possible_actions,
            possible_actions_mask=possible_actions_mask,
            possible_next_actions=possible_next_actions,
            possible_next_actions_mask=possible_next_actions_mask,
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


MAKER_MAP = {
    rlt.DiscreteDqnInput: DiscreteDqnInputMaker,
    rlt.PolicyNetworkInput: PolicyNetworkInputMaker,
    rlt.MemoryNetworkInput: MemoryNetworkInputMaker,
    rlt.ParametricDqnInput: ParametricDqnInputMaker,
    rlt.SlateQInput: SlateQInputMaker,
}
