#!/usr/bin/env python3

import numpy as np
import torch
from caffe2.python import workspace
from ml.rl.test.gym.open_ai_gym_environment import ModelType
from ml.rl.training.training_data_page import TrainingDataPage


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

    def sample_memories(self, batch_size, model_type):
        """
        Samples transitions from replay memory uniformly at random.

        :param batch_size: Number of sampled transitions to return.
        :param model_type: Model type (discrete, parametric).
        """
        cols = [[], [], [], [], [], [], [], [], []]
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)

        possible_next_actions_lengths = torch.tensor(cols[7], dtype=torch.int32)
        next_states = torch.tensor(cols[3], dtype=torch.float32)

        if model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
            possible_next_actions = []
            for pna_matrix in cols[6]:
                for row in pna_matrix:
                    possible_next_actions.append(row)

            tiled_states = torch.from_numpy(
                np.repeat(
                    next_states.numpy(), possible_next_actions_lengths.numpy(), axis=0
                )
            )
            possible_next_actions = torch.tensor(
                possible_next_actions, dtype=torch.float32
            )
            possible_next_actions_state_concat = torch.cat(
                (tiled_states, possible_next_actions), dim=1
            )
        else:
            if cols[6] is None or cols[6][0] is None:
                possible_next_actions = None
            else:
                possible_next_actions = torch.tensor(cols[6], dtype=torch.float32)
            possible_next_actions_state_concat = None

        return TrainingDataPage(
            states=torch.tensor(cols[0], dtype=torch.float32),
            actions=torch.tensor(cols[1], dtype=torch.float32),
            propensities=None,
            rewards=torch.tensor(cols[2], dtype=torch.float32).reshape(-1, 1),
            next_states=torch.tensor(cols[3], dtype=torch.float32),
            next_actions=torch.tensor(cols[4], dtype=torch.float32),
            possible_next_actions=possible_next_actions,
            episode_values=None,
            not_terminals=torch.from_numpy(
                np.logical_not(np.array(cols[5]), dtype=np.bool).astype(np.int32)
            ).reshape(-1, 1),
            time_diffs=torch.tensor(cols[8], dtype=torch.int32).reshape(-1, 1),
            possible_next_actions_lengths=possible_next_actions_lengths,
            possible_next_actions_state_concat=possible_next_actions_state_concat,
        )

    def insert_into_memory(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        terminal,
        possible_next_actions,
        possible_next_actions_lengths,
        time_diff,
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
            possible_next_actions_lengths,
            time_diff,
        )

        if self.memory_num < self.max_replay_memory_size:
            self.replay_memory.append(item)
        elif self.memory_num >= self.skip_insert_until:
            p = float(self.max_replay_memory_size) / self.memory_num
            self.skip_insert_until += np.random.geometric(p)
            rand_index = np.random.randint(self.max_replay_memory_size)
            self.replay_memory[rand_index] = item
        self.memory_num += 1
