#!/usr/bin/env python3

import numpy as np
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

        possible_next_actions_lengths = np.array(cols[7], dtype=np.int32)
        next_states = np.array(cols[3], dtype=np.float32)

        if model_type in (
            ModelType.PARAMETRIC_ACTION.value,
            ModelType.PYTORCH_PARAMETRIC_DQN.value,
        ):
            possible_next_actions = []
            for pna_matrix in cols[6]:
                for row in pna_matrix:
                    possible_next_actions.append(row)

            tiled_states = np.repeat(next_states, possible_next_actions_lengths, axis=0)
            possible_next_actions = np.array(possible_next_actions, dtype=np.float32)
            next_state_pnas_concat = np.concatenate(
                (tiled_states, possible_next_actions), axis=1
            )
        else:
            possible_next_actions = np.array(cols[6], dtype=np.float32)
            next_state_pnas_concat = None

        return TrainingDataPage(
            states=np.array(cols[0], dtype=np.float32),
            actions=np.array(cols[1], dtype=np.float32),
            propensities=None,
            rewards=np.array(cols[2], dtype=np.float32),
            next_states=np.array(cols[3], dtype=np.float32),
            next_actions=np.array(cols[4], dtype=np.float32),
            possible_next_actions=possible_next_actions,
            episode_values=None,
            not_terminals=np.logical_not(np.array(cols[5]), dtype=np.bool),
            time_diffs=np.array(cols[8], dtype=np.int32),
            possible_next_actions_lengths=possible_next_actions_lengths,
            next_state_pnas_concat=next_state_pnas_concat,
        )

    def sample_and_load_training_data_c2(self, num_samples, model_type):
        """
        Loads and preprocesses shuffled, transformed transitions from
        replay memory into the training net.

        :param num_samples: Number of transitions to sample from replay memory.
        :param model_type: Model type (discrete, parametric).
        """
        tdp = self.sample_memories(num_samples, model_type)
        workspace.FeedBlob("states", tdp.states)
        workspace.FeedBlob("actions", tdp.actions)
        workspace.FeedBlob("rewards", tdp.rewards.reshape(-1, 1))
        workspace.FeedBlob("next_states", tdp.next_states)
        workspace.FeedBlob("not_terminals", tdp.not_terminals.reshape(-1, 1))
        workspace.FeedBlob("time_diff", tdp.time_diffs.reshape(-1, 1))
        workspace.FeedBlob("next_actions", tdp.next_actions)
        workspace.FeedBlob("possible_next_actions", tdp.possible_next_actions)
        workspace.FeedBlob(
            "possible_next_actions_lengths", tdp.possible_next_actions_lengths
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
