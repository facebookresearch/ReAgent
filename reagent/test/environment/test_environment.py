#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

from reagent.test.gridworld.gridworld import Gridworld
from reagent.test.gridworld.gridworld_continuous import GridworldContinuous
from reagent.test.gym.open_ai_gym_environment import OpenAIGymEnvironment


logger = logging.getLogger(__name__)


class TestEnvironment(unittest.TestCase):
    def test_gridworld_generate_samples(self):
        env = Gridworld()
        num_samples = 1000
        num_steps = 5
        samples = env.generate_samples(
            num_samples,
            epsilon=1.0,
            discount_factor=0.9,
            multi_steps=num_steps,
            include_shorter_samples_at_start=True,
            include_shorter_samples_at_end=True,
        )
        self._check_samples(samples, num_samples, num_steps, False)

    def test_gridworld_continuous_generate_samples(self):
        env = GridworldContinuous()
        num_samples = 1000
        num_steps = 5
        samples = env.generate_samples(
            num_samples,
            epsilon=1.0,
            discount_factor=0.9,
            multi_steps=num_steps,
            include_shorter_samples_at_start=True,
            include_shorter_samples_at_end=True,
        )
        self._check_samples(samples, num_samples, num_steps, True)

    def test_open_ai_gym_generate_samples_multi_step(self):
        env = OpenAIGymEnvironment(
            "CartPole-v0",
            epsilon=1.0,  # take random actions to collect training data
            softmax_policy=False,
            gamma=0.9,
        )
        num_samples = 1000
        num_steps = 5
        samples = env.generate_random_samples(
            num_samples,
            use_continuous_action=True,
            epsilon=1.0,
            multi_steps=num_steps,
            include_shorter_samples_at_start=True,
            include_shorter_samples_at_end=True,
        )
        self._check_samples(samples, num_samples, num_steps, True)

    def _check_samples(self, samples, num_samples, num_steps, continuous_action):
        if continuous_action:
            assertActionEqual = self.assertDictEqual
        else:
            assertActionEqual = self.assertEqual
        for i in range(num_samples):
            if samples.terminals[i][0]:
                break
            self.assertEqual(samples.mdp_ids[i], samples.mdp_ids[i + 1])
            if i >= num_steps - 1:
                self.assertEqual(
                    samples.sequence_numbers[i] + 1, samples.sequence_numbers[i + 1]
                )
                for j in range(len(samples.terminals[i])):
                    self.assertEqual(samples.rewards[i][j], samples.rewards[i + j][0])
                    self.assertDictEqual(
                        samples.next_states[i][j], samples.next_states[i + j][0]
                    )
                    assertActionEqual(
                        samples.next_actions[i][j], samples.next_actions[i + j][0]
                    )
                    self.assertEqual(
                        samples.terminals[i][j], samples.terminals[i + j][0]
                    )
                    self.assertListEqual(
                        samples.possible_next_actions[i][j],
                        samples.possible_next_actions[i + j][0],
                    )
                    if samples.terminals[i][j]:
                        continue
                    self.assertDictEqual(
                        samples.next_states[i][j], samples.states[i + j + 1]
                    )
                    assertActionEqual(
                        samples.next_actions[i][j], samples.actions[i + j + 1]
                    )
                    self.assertListEqual(
                        samples.possible_next_actions[i][j],
                        samples.possible_actions[i + j + 1],
                    )
            else:
                self.assertEqual(
                    samples.sequence_numbers[i], samples.sequence_numbers[i + 1]
                )
                for j in range(1, len(samples.terminals[i])):
                    self.assertEqual(
                        samples.rewards[i][j], samples.rewards[i + num_steps - 1][0]
                    )
                    self.assertDictEqual(
                        samples.next_states[i][j],
                        samples.next_states[num_steps - 1 + j][0],
                    )
                    assertActionEqual(
                        samples.next_actions[i][j],
                        samples.next_actions[num_steps - 1 + j][0],
                    )
                    self.assertEqual(
                        samples.terminals[i][j], samples.terminals[num_steps - 1 + j][0]
                    )
                    self.assertListEqual(
                        samples.possible_next_actions[i][j],
                        samples.possible_next_actions[num_steps - 1 + j][0],
                    )
                    if samples.terminals[i][j]:
                        continue
                    self.assertDictEqual(
                        samples.next_states[i][j], samples.states[num_steps + j]
                    )
                    assertActionEqual(
                        samples.next_actions[i][j], samples.actions[num_steps + j]
                    )
                    self.assertListEqual(
                        samples.possible_next_actions[i][j],
                        samples.possible_actions[num_steps + j],
                    )

        single_step_samples = samples.to_single_step()
        for i in range(num_samples):
            if single_step_samples.terminals[i] is True:
                break
            self.assertEqual(single_step_samples.mdp_ids[i], samples.mdp_ids[i])
            self.assertEqual(
                single_step_samples.sequence_numbers[i], samples.sequence_numbers[i]
            )
            self.assertDictEqual(single_step_samples.states[i], samples.states[i])
            assertActionEqual(single_step_samples.actions[i], samples.actions[i])
            self.assertEqual(
                single_step_samples.action_probabilities[i],
                samples.action_probabilities[i],
            )
            self.assertEqual(single_step_samples.rewards[i], samples.rewards[i][0])
            self.assertListEqual(
                single_step_samples.possible_actions[i], samples.possible_actions[i]
            )
            self.assertDictEqual(
                single_step_samples.next_states[i], samples.next_states[i][0]
            )
            assertActionEqual(
                single_step_samples.next_actions[i], samples.next_actions[i][0]
            )
            self.assertEqual(single_step_samples.terminals[i], samples.terminals[i][0])
            self.assertListEqual(
                single_step_samples.possible_next_actions[i],
                samples.possible_next_actions[i][0],
            )
