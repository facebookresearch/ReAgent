#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import dataclasses
import logging
import random
from typing import Optional, Union

import numpy as np
import torch
from reagent.test.gym.open_ai_gym_environment import ModelType
from reagent.torch_utils import stack
from reagent.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MemoryBuffer:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_action: torch.Tensor
    terminal: torch.Tensor
    possible_next_actions: Optional[torch.Tensor]
    possible_next_actions_mask: Optional[torch.Tensor]
    possible_actions: Optional[torch.Tensor]
    possible_actions_mask: Optional[torch.Tensor]
    time_diff: torch.Tensor
    policy_id: torch.Tensor
    propensity: Optional[torch.Tensor]
    next_propensity: Optional[torch.Tensor]
    reward_mask: Optional[torch.Tensor]

    @torch.no_grad()
    def slice(self, indices):
        return MemoryBuffer(
            state=self.state[indices],
            action=self.action[indices],
            reward=self.reward[indices],
            reward_mask=self.reward_mask[indices]
            if self.reward_mask is not None
            else None,
            next_state=self.next_state[indices],
            next_action=self.next_action[indices],
            terminal=self.terminal[indices],
            possible_next_actions=self.possible_next_actions[indices]
            if self.possible_next_actions is not None
            else None,
            possible_next_actions_mask=self.possible_next_actions_mask[indices]
            if self.possible_next_actions_mask is not None
            else None,
            possible_actions=self.possible_actions[indices]
            if self.possible_actions is not None
            else None,
            possible_actions_mask=self.possible_actions_mask[indices]
            if self.possible_actions_mask is not None
            else None,
            time_diff=self.time_diff[indices],
            policy_id=self.policy_id[indices],
            propensity=self.propensity[indices]
            if self.propensity is not None
            else None,
            next_propensity=self.next_propensity[indices]
            if self.next_propensity is not None
            else None,
        )

    @torch.no_grad()
    def insert_at(
        self,
        idx: int,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: Union[float, torch.Tensor],
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        terminal: bool,
        possible_next_actions: Optional[torch.Tensor],
        possible_next_actions_mask: Optional[torch.Tensor],
        time_diff: float,
        possible_actions: Optional[torch.Tensor],
        possible_actions_mask: Optional[torch.Tensor],
        policy_id: int,
        propensity: Optional[torch.Tensor],
        next_propensity: Optional[torch.Tensor],
        reward_mask: Optional[torch.Tensor],
    ):
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        if self.reward_mask is not None:
            # pyre-fixme[16]: `Optional` has no attribute `__setitem__`.
            self.reward_mask[idx] = reward_mask
        self.next_state[idx] = next_state
        self.next_action[idx] = next_action
        self.terminal[idx] = terminal
        if self.possible_actions is not None:
            self.possible_actions[idx] = possible_actions
        if self.possible_actions_mask is not None:
            self.possible_actions_mask[idx] = possible_actions_mask
        if self.possible_next_actions is not None:
            self.possible_next_actions[idx] = possible_next_actions
        if self.possible_next_actions_mask is not None:
            self.possible_next_actions_mask[idx] = possible_next_actions_mask
        self.time_diff[idx] = time_diff
        self.policy_id[idx] = policy_id
        if self.propensity is not None:
            self.propensity[idx] = propensity
        if self.next_propensity is not None:
            self.next_propensity[idx] = next_propensity

    @classmethod
    def create(
        cls,
        max_size: int,
        state_dim: int,
        action_dim: int,
        max_possible_actions: Optional[int],
        has_possble_actions: bool,
        has_propensity: bool,
        reward_dim: Optional[int],
    ):
        return cls(
            state=torch.zeros((max_size, state_dim)),
            action=torch.zeros((max_size, action_dim)),
            reward=torch.zeros((max_size, reward_dim if reward_dim is not None else 1)),
            reward_mask=torch.zeros((max_size, reward_dim), dtype=torch.bool)
            if reward_dim is not None
            else None,
            next_state=torch.zeros((max_size, state_dim)),
            next_action=torch.zeros((max_size, action_dim)),
            terminal=torch.zeros((max_size, 1), dtype=torch.uint8),
            possible_next_actions=torch.zeros(
                (max_size, max_possible_actions, action_dim)
            )
            if has_possble_actions
            else None,
            possible_next_actions_mask=torch.zeros(
                (max_size, max_possible_actions), dtype=torch.bool
            )
            if max_possible_actions
            else None,
            possible_actions=torch.zeros((max_size, max_possible_actions, action_dim))
            if has_possble_actions
            else None,
            possible_actions_mask=torch.zeros(
                (max_size, max_possible_actions), dtype=torch.bool
            )
            if max_possible_actions
            else None,
            time_diff=torch.zeros((max_size, 1)),
            policy_id=torch.zeros((max_size, 1), dtype=torch.long),
            propensity=torch.zeros(
                (
                    max_size,
                    max_possible_actions if max_possible_actions is not None else 1,
                )
            )
            if has_propensity
            else None,
            next_propensity=torch.zeros(
                (
                    max_size,
                    max_possible_actions if max_possible_actions is not None else 1,
                )
            )
            if has_propensity
            else None,
        )


class OpenAIGymMemoryPool:
    def __init__(self, max_replay_memory_size: int):
        """
        Creates an OpenAIGymMemoryPool object.

        :param max_replay_memory_size: Upper bound on the number of transitions
            to store in replay memory.
        """
        self.max_replay_memory_size = max_replay_memory_size
        self.memory_num = 0

        # Not initializing in the beginning because we don't know the shapes
        self.memory_buffer: Optional[MemoryBuffer] = None

    @property
    def size(self):
        return min(self.memory_num, self.max_replay_memory_size)

    @property
    def state_dim(self):
        assert self.memory_buffer is not None
        return self.memory_buffer.state.shape[1]

    @property
    def action_dim(self):
        assert self.memory_buffer is not None
        return self.memory_buffer.action.shape[1]

    def sample_memories(self, batch_size, model_type, chunk=None):
        """
        Samples transitions from replay memory uniformly at random by default
        or pass chunk for deterministic sample.

        *Note*: 1-D vectors such as state & action get stacked to make a 2-D
        matrix, while a 2-D matrix such as possible_actions (in the parametric
        case) get concatenated to make a bigger 2-D matrix

        :param batch_size: Number of sampled transitions to return.
        :param model_type: Model type (discrete, parametric).
        :param chunk: Index of chunk of data (for deterministic sampling).
        """
        if chunk is None:
            indices = torch.randint(0, self.size, size=(batch_size,))
        else:
            start_idx = chunk * batch_size
            end_idx = start_idx + batch_size
            indices = range(start_idx, end_idx)

        memory = self.memory_buffer.slice(indices)

        states = memory.state
        next_states = memory.next_state

        assert states.dim() == 2
        assert next_states.dim() == 2

        if model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
            num_possible_actions = memory.possible_actions_mask.shape[1]

            actions = memory.action
            next_actions = memory.next_action

            tiled_states = states.repeat(1, num_possible_actions).reshape(
                -1, states.shape[1]
            )
            possible_actions = memory.possible_actions.reshape(-1, actions.shape[1])
            possible_actions_state_concat = torch.cat(
                (tiled_states, possible_actions), dim=1
            )
            possible_actions_mask = memory.possible_actions_mask

            tiled_next_states = next_states.repeat(1, num_possible_actions).reshape(
                -1, next_states.shape[1]
            )
            possible_next_actions = memory.possible_next_actions.reshape(
                -1, actions.shape[1]
            )
            possible_next_actions_state_concat = torch.cat(
                (tiled_next_states, possible_next_actions), dim=1
            )
            possible_next_actions_mask = memory.possible_next_actions_mask
        else:
            possible_actions = None
            possible_actions_state_concat = None
            possible_next_actions = None
            possible_next_actions_state_concat = None
            possible_next_actions_mask = memory.possible_next_actions_mask
            possible_actions_mask = memory.possible_actions_mask

            actions = memory.action
            next_actions = memory.next_action

            assert len(actions.size()) == 2
            assert len(next_actions.size()) == 2

        rewards = memory.reward
        rewards_mask = memory.reward_mask
        not_terminal = 1 - memory.terminal
        time_diffs = memory.time_diff
        propensities = memory.propensity
        next_propensities = memory.next_propensity

        return TrainingDataPage(
            states=states,
            actions=actions,
            propensities=propensities,
            next_propensities=next_propensities,
            rewards=rewards,
            rewards_mask=rewards_mask,
            next_states=next_states,
            next_actions=next_actions,
            not_terminal=not_terminal,
            time_diffs=time_diffs,
            possible_actions_mask=possible_actions_mask,
            possible_actions_state_concat=possible_actions_state_concat,
            possible_next_actions_mask=possible_next_actions_mask,
            possible_next_actions_state_concat=possible_next_actions_state_concat,
        )

    def insert_into_memory(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: Union[float, torch.Tensor],
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        terminal: bool,
        possible_next_actions: Optional[torch.Tensor],
        possible_next_actions_mask: Optional[torch.Tensor],
        time_diff: float,
        possible_actions: Optional[torch.Tensor],
        possible_actions_mask: Optional[torch.Tensor],
        policy_id: int,
        propensity: Optional[torch.Tensor] = None,
        next_propensity: Optional[torch.Tensor] = None,
        reward_mask: Optional[torch.Tensor] = None,
    ):
        """
        Inserts transition into replay memory in such a way that retrieving
        transitions uniformly at random will be equivalent to reservoir sampling.
        """

        if self.memory_buffer is None:
            assert state.shape == next_state.shape
            assert len(state.shape) == 1
            assert action.shape == next_action.shape
            assert len(action.shape) == 1
            if possible_actions_mask is not None:
                assert possible_next_actions_mask is not None
                assert possible_actions_mask.shape == possible_next_actions_mask.shape
                assert len(possible_actions_mask.shape) == 1
                max_possible_actions = possible_actions_mask.shape[0]
            else:
                max_possible_actions = None

            assert (possible_actions is not None) == (possible_next_actions is not None)
            assert (propensity is None) == (next_propensity is None)
            reward_dim = None
            if isinstance(reward, torch.Tensor):
                assert reward_mask is not None
                assert reward.shape == reward_mask.shape
                assert len(reward.shape) == 1
                reward_dim = reward.shape[0]
            else:
                assert reward_mask is None

            self.memory_buffer = MemoryBuffer.create(
                max_size=self.max_replay_memory_size,
                state_dim=state.shape[0],
                action_dim=action.shape[0],
                max_possible_actions=max_possible_actions,
                has_possble_actions=possible_actions is not None,
                has_propensity=propensity is not None,
                reward_dim=reward_dim,
            )

        insert_idx = None
        if self.memory_num < self.max_replay_memory_size:
            insert_idx = self.memory_num
        else:
            rand_idx = torch.randint(0, self.memory_num, size=(1,)).item()
            if rand_idx < self.max_replay_memory_size:
                insert_idx = rand_idx

        if insert_idx is not None:
            # pyre-fixme[16]: `Optional` has no attribute `insert_at`.
            self.memory_buffer.insert_at(
                insert_idx,
                state,
                action,
                reward,
                next_state,
                next_action,
                terminal,
                possible_next_actions,
                possible_next_actions_mask,
                time_diff,
                possible_actions,
                possible_actions_mask,
                policy_id,
                propensity,
                next_propensity,
                reward_mask,
            )
        self.memory_num += 1
