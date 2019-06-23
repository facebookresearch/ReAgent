#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import random
from typing import Optional

import numpy as np
import torch
from caffe2.python import workspace
from ml.rl.test.gym.open_ai_gym_environment import ModelType
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class OpenAIGymMemoryPool:
    def __init__(self, max_replay_memory_size):
        """
        Creates an OpenAIGymMemoryPool object.

        :param max_replay_memory_size: Upper bound on the number of transitions
            to store in replay memory.
        """
        self.replay_memory = []
        self.max_replay_memory_size = max_replay_memory_size
        self.memory_num = 0
        self.skip_insert_until = self.max_replay_memory_size

    @property
    def size(self):
        return len(self.replay_memory)

    def shuffle(self):
        random.shuffle(self.replay_memory)

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
        cols = [[], [], [], [], [], [], [], [], [], [], [], []]

        if chunk is None:
            indices = torch.randperm(len(self.replay_memory))[:batch_size]
        else:
            start_idx = chunk * batch_size
            end_idx = start_idx + batch_size
            indices = range(start_idx, end_idx)

        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)

        states = torch.stack(cols[0])
        next_states = torch.stack(cols[3])

        assert states.dim() == 2
        assert next_states.dim() == 2

        if model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
            num_possible_actions = len(cols[7][0])

            actions = torch.stack(cols[1])
            next_actions = torch.stack(cols[4])

            tiled_states = states.repeat(1, num_possible_actions).reshape(
                -1, states.shape[1]
            )
            possible_actions = torch.cat(cols[8])
            possible_actions_state_concat = torch.cat(
                (tiled_states, possible_actions), dim=1
            )
            possible_actions_mask = torch.stack(cols[9])

            possible_next_actions = torch.cat(cols[6])

            tiled_next_states = next_states.repeat(1, num_possible_actions).reshape(
                -1, next_states.shape[1]
            )

            possible_next_actions_state_concat = torch.cat(
                (tiled_next_states, possible_next_actions), dim=1
            )
            possible_next_actions_mask = torch.stack(cols[7])
        else:
            possible_actions = None
            possible_actions_state_concat = None
            possible_next_actions = None
            possible_next_actions_state_concat = None
            if cols[7] is None or cols[7][0] is None:
                possible_next_actions_mask = None
            else:
                possible_next_actions_mask = torch.stack(cols[7])
            if cols[9] is None or cols[9][0] is None:
                possible_actions_mask = None
            else:
                possible_actions_mask = torch.stack(cols[9])

            actions = torch.stack(cols[1])
            next_actions = torch.stack(cols[4])

            assert len(actions.size()) == 2
            assert len(next_actions.size()) == 2

        return TrainingDataPage(
            states=states,
            actions=actions,
            propensities=None,
            rewards=torch.tensor(cols[2], dtype=torch.float32).reshape(-1, 1),
            next_states=next_states,
            next_actions=next_actions,
            not_terminal=(1 - torch.tensor(cols[5], dtype=torch.int32)).reshape(-1, 1),
            time_diffs=torch.tensor(cols[10], dtype=torch.int32).reshape(-1, 1),
            possible_actions_mask=possible_actions_mask,
            possible_actions_state_concat=possible_actions_state_concat,
            possible_next_actions_mask=possible_next_actions_mask,
            possible_next_actions_state_concat=possible_next_actions_state_concat,
        )

    def insert_into_memory(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        terminal: bool,
        possible_next_actions: Optional[torch.Tensor],
        possible_next_actions_mask: Optional[torch.Tensor],
        time_diff: float,
        possible_actions: Optional[torch.Tensor],
        possible_actions_mask: Optional[torch.Tensor],
        policy_id: str,
    ):
        """
        Inserts transition into replay memory in such a way that retrieving
        transitions uniformly at random will be equivalent to reservoir sampling.
        """
        item = (
            state,
            action,
            reward,
            next_state,
            next_action,
            terminal,
            possible_next_actions,
            possible_next_actions_mask,
            possible_actions,
            possible_actions_mask,
            time_diff,
            policy_id,
        )

        if self.memory_num < self.max_replay_memory_size:
            self.replay_memory.append(item)
        elif self.memory_num >= self.skip_insert_until:
            p = float(self.max_replay_memory_size) / self.memory_num
            self.skip_insert_until += np.random.geometric(p)
            rand_index = np.random.randint(self.max_replay_memory_size)
            self.replay_memory[rand_index] = item
        self.memory_num += 1
