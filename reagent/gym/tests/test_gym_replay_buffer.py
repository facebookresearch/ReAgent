#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging

import numpy.testing as npt
from reagent.core.parameters import ProblemDomain
from reagent.gym.envs import Gym
from reagent.gym.envs.wrappers.simple_minigrid import SimpleObsWrapper
from reagent.gym.utils import create_df_from_replay_buffer
from reagent.preprocessing.sparse_to_dense import PythonSparseToDenseProcessor

from reagent.test.base.horizon_test_base import HorizonTestBase

logger = logging.getLogger(__name__)


class TestEnv(SimpleObsWrapper):
    """
    Wrap Gym environment in TestEnv to save the MiniGrid's
    observation, action, reward and terminal in a list so that
    we can check if replay buffer is working correctly
    """

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        # mdp_id, sequence_number, state, action, reward, terminal
        self.sart = []
        self.mdp_id = -1
        self.sequence_number = 0

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)

    def reset(self, **kwargs):
        self.mdp_id += 1
        self.sequence_number = 0
        res = self.env.reset(**kwargs)
        self.sart.append([self.mdp_id, self.sequence_number, res, None, None, None])
        return res

    def step(self, action):
        res = self.env.step(action)
        (
            _,
            _,
            last_state,
            last_action,
            last_reward,
            last_terminal,
        ) = self.sart[-1]
        assert (
            last_state is not None
            and last_action is None
            and last_reward is None
            and last_terminal is None
        )
        next_state, reward, terminal, _ = res
        self.sart[-1][3] = action
        self.sart[-1][4] = reward
        self.sart[-1][5] = terminal
        self.sequence_number += 1
        self.sart.append(
            [self.mdp_id, self.sequence_number, next_state, None, None, None]
        )
        return res


class TestGymReplayBuffer(HorizonTestBase):
    def test_create_df_from_replay_buffer(self):
        env_name = "MiniGrid-Empty-5x5-v0"
        env = Gym(env_name=env_name)
        state_dim = env.observation_space.shape[0]
        # Wrap env in TestEnv
        env = TestEnv(env)
        problem_domain = ProblemDomain.DISCRETE_ACTION
        DATASET_SIZE = 1000
        multi_steps = None
        DS = "2021-09-16"

        # Generate data
        df = create_df_from_replay_buffer(
            env=env,
            problem_domain=problem_domain,
            desired_size=DATASET_SIZE,
            multi_steps=multi_steps,
            ds=DS,
            shuffle_df=False,
        )
        self.assertEqual(len(df), DATASET_SIZE)

        # Check data
        preprocessor = PythonSparseToDenseProcessor(list(range(state_dim)))
        for idx, row in df.iterrows():
            df_mdp_id = row["mdp_id"]
            env_mdp_id = str(env.sart[idx][0])
            self.assertEqual(df_mdp_id, env_mdp_id)

            df_seq_num = row["sequence_number"]
            env_seq_num = env.sart[idx][1]
            self.assertEqual(df_seq_num, env_seq_num)

            df_state = preprocessor.process([row["state_features"]])[0][0].numpy()
            env_state = env.sart[idx][2]
            npt.assert_array_equal(df_state, env_state)

            df_action = row["action"]
            env_action = str(env.sart[idx][3])
            self.assertEqual(df_action, env_action)

            df_terminal = row["next_action"] == ""
            env_terminal = env.sart[idx][5]
            self.assertEqual(df_terminal, env_terminal)
            if not df_terminal:
                df_reward = float(row["reward"])
                env_reward = float(env.sart[idx][4])
                npt.assert_allclose(df_reward, env_reward)

                df_next_state = preprocessor.process([row["next_state_features"]])[0][
                    0
                ].numpy()
                env_next_state = env.sart[idx + 1][2]
                npt.assert_array_equal(df_next_state, env_next_state)

                df_next_action = row["next_action"]
                env_next_action = str(env.sart[idx + 1][3])
                self.assertEqual(df_next_action, env_next_action)
            else:
                del env.sart[idx + 1]
