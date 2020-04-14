#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.simulators.recsim import RecSim


class RandomPolicy:
    def __init__(self, m):
        self.w = torch.ones(1, m) / m
        self.generator = torch._C.Generator()
        self.generator.manual_seed(10101)

    def __call__(self, num_active_users, k):
        w = self.w.repeat(num_active_users, 1)
        action = torch.multinomial(w, k, generator=self.generator)
        return action


class TestRecsim(unittest.TestCase):
    def test_default(self):
        recsim = RecSim()
        policy = RandomPolicy(recsim.m)

        cum_reward = 0

        for _i in range(10000):
            active_user_ids, users, candidates = recsim.obs()
            action = policy(active_user_ids.shape[0], recsim.k)
            reward, user_choice, interest, num_active_users = recsim.step(action)
            cum_reward += reward.sum().item()
            if num_active_users == 0:
                break
        else:
            self.fail("Running too long")

        self.assertEqual(646144, cum_reward)
