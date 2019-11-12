#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import ml.rl.types as rlt
import numpy as np
import torch
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.simulators.recsim import RecSim
from ml.rl.test.base.horizon_test_base import HorizonTestBase
from ml.rl.test.gym.open_ai_gym_environment import ModelType
from ml.rl.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool
from ml.rl.training.slate_q_trainer import SlateQTrainer, SlateQTrainerParameters


class TestSlateQTrainer(HorizonTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        np.random.seed(0)

    def test_slate_q_trainer(self):
        recsim = RecSim(num_users=10)

        # Build memory pool with random policy

        memory_pool = OpenAIGymMemoryPool(10000000)

        prev_obs = None
        prev_action = None
        prev_user_choice = None
        prev_reward = None
        # TODO: Record interest as propensity in the replay buffer

        random_reward = 0

        while True:
            obs = recsim.obs()
            active_user_idxs, user_features, candidate_features = obs
            item_idxs = torch.multinomial(
                torch.ones(active_user_idxs.shape[0], recsim.m), recsim.k
            )
            # TODO: Record propensity
            reward, user_choice, interest, num_alive = recsim.step(item_idxs)

            random_reward += reward.sum().item()

            action_features = recsim.select(
                candidate_features, item_idxs, True
            ).as_vector()

            if prev_obs is not None:
                prev_active_user_idxs, prev_user_features, prev_candidate_features = (
                    prev_obs
                )
                i, j = 0, 0
                while i < len(prev_active_user_idxs):
                    mdp_id = prev_active_user_idxs[i]
                    state = prev_user_features[i]
                    possible_actions = prev_action[i]
                    action = possible_actions[prev_user_choice[i]].view(-1)
                    possible_actions_mask = (
                        torch.arange(recsim.k + 1) == prev_user_choice[i]
                    )

                    if j < len(active_user_idxs) and mdp_id == active_user_idxs[j]:
                        # not terminated
                        terminal = False
                        next_state = user_features[j]
                        possible_next_actions = action_features[j]
                        next_action = possible_next_actions[user_choice[j]].view(-1)
                        j += 1
                    else:
                        terminal = True
                        next_state = torch.zeros_like(state)
                        possible_next_actions = torch.zeros_like(action)
                        next_action = possible_next_actions[0].view(-1)

                    # This doesn't matter
                    possible_next_actions_mask = torch.ones(
                        recsim.k + 1, dtype=torch.uint8
                    )

                    memory_pool.insert_into_memory(
                        state=state,
                        action=action,
                        reward=prev_reward[i].item(),
                        next_state=next_state,
                        next_action=next_action,
                        terminal=terminal,
                        possible_next_actions=possible_next_actions,
                        possible_next_actions_mask=possible_next_actions_mask,
                        time_diff=1.0,
                        possible_actions=possible_actions,
                        possible_actions_mask=possible_actions_mask,
                        policy_id=1,
                    )

                    i += 1

            prev_obs = obs
            prev_action = action_features
            prev_user_choice = user_choice
            prev_reward = reward

            if num_alive == 0:
                break

        self.assertEqual(random_reward, 1384.0)

        # Train a model
        q_network = FullyConnectedParametricDQN(
            state_dim=state.shape[0],
            action_dim=action.shape[0],
            sizes=[64, 32],
            activations=["relu", "relu"],
        )

        q_network = q_network.eval()
        untrained_policy_reward = self._rollout_policy(q_network, recsim)
        self.assertEqual(untrained_policy_reward, 1212.0)
        q_network = q_network.train()

        q_network_target = q_network.get_target_network()
        parameters = SlateQTrainerParameters()
        trainer = SlateQTrainer(q_network, q_network_target, parameters)

        for _i in range(1000):
            tdp = memory_pool.sample_memories(
                128, model_type=ModelType.PYTORCH_PARAMETRIC_DQN.value
            )
            training_batch = tdp.as_slate_q_training_batch()
            trainer.train(training_batch)

        # TODO: Test the policy, it should be better than random
        q_network = q_network.eval()
        trained_policy_reward = self._rollout_policy(q_network, recsim)

        print(
            f"Reward; random: {random_reward}; untrained: {untrained_policy_reward}; "
            f"trained: {trained_policy_reward}"
        )

        self.assertGreater(trained_policy_reward, untrained_policy_reward)
        self.assertGreater(trained_policy_reward, random_reward)
        self.assertEqual(trained_policy_reward, 1416.0)

    def _rollout_policy(self, q_network, recsim):
        recsim.reset()
        policy_reward = 0
        while True:
            obs = recsim.obs()
            active_user_idxs, user_features, candidate_features = obs

            tiled_user_features = user_features.repeat_interleave(recsim.m, dim=0)
            candidate_features = candidate_features.as_vector()
            action_dim = candidate_features.shape[2]
            flatten_candidate_features = candidate_features.view(-1, action_dim)

            q_network_input = rlt.PreprocessedStateAction.from_tensors(
                state=tiled_user_features, action=flatten_candidate_features
            )
            q_values = q_network(q_network_input).q_value.view(-1, recsim.m)

            top_q_values, item_idxs = torch.topk(q_values, recsim.k, dim=1)

            reward, user_choice, interest, num_alive = recsim.step(item_idxs)

            policy_reward += reward.sum().item()

            if num_alive == 0:
                break
        return policy_reward
