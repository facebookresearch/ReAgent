#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import random

import numpy as np
import unittest

from ml.rl.training.discrete_action_trainer import \
    DiscreteActionTrainer
from ml.rl.thrift.core.ttypes import \
    RLParameters, TrainingParameters, \
    DiscreteActionModelParameters, ActionBudget
from ml.rl.training.limited_discrete_action_trainer import \
    LimitedActionDiscreteActionTrainer
from ml.rl.test.gridworld.gridworld_base import \
    DISCOUNT
from ml.rl.test.gridworld.limited_action_gridworld import \
    LimitedActionGridworld

IterationResult = collections.namedtuple(
    'IterationResult', [
        'states', 'actions', 'rewards', 'next_states', 'next_actions',
        'is_terminals', 'possible_next_actions', 'lengths', 'cheat_ratio',
        'current_state'
    ]
)


def _build_policy(env, predictor, epsilon):
    eye = np.eye(env.num_states)
    q_values = predictor.predict(
        {str(i): eye[i]
         for i in range(env.num_states)}
    )
    policy_vector = [
        env.ACTIONS[np.argmax([q_values[action][i] for action in env.ACTIONS])]
        for i in range(env.num_states)
    ]

    def policy(state) -> str:
        if np.random.random() < epsilon:
            return np.random.choice(env.ACTIONS)
        else:
            return policy_vector[state]

    return policy


def _collect_samples(env, policy, num_steps, initial_state):
    states = []
    actions = []
    rewards = []
    next_states = []
    next_actions = []
    possible_next_actions = []
    is_terminals = []
    lengths = []
    num_cheats = []

    num_cheat = 0
    solved_in = 0

    next_state = initial_state
    next_action = policy(next_state)
    for _ in range(num_steps):
        state = next_state
        action = next_action
        next_state, reward, is_terminal, _ = env.step(action)
        reward -= 0.01
        states.append({str(state): 1.0})
        actions.append(action)
        next_states.append({str(next_state): 1.0})
        next_action = policy(next_state)
        next_actions.append(next_action)
        possible_next_actions.append(env.possible_next_actions(state))
        is_terminals.append(is_terminal)
        if action == 'C':
            num_cheat += 1
        solved_in += 1
        rewards.append(reward)
        if is_terminal:
            lengths.append(solved_in)
            num_cheats.append(num_cheat)
            next_state = env.reset()
            next_action = policy(next_state)
            solved_in = 0
            num_cheat = 0
    cheat_ratio = np.sum(np.array(actions) == 'C') / len(actions)
    return IterationResult(
        states, actions, rewards, next_states, next_actions, is_terminals,
        possible_next_actions, lengths, cheat_ratio, next_state
    )


class TestLimitedActionGridworld(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        np.random.seed(0)
        random.seed(0)

        self._env = LimitedActionGridworld()
        self._rl_parameters = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.05,
            reward_burnin=10,
            maxq_learning=False,
        )
        self._rl_parameters_maxq = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.05,
            reward_burnin=10,
            maxq_learning=True,
        )
        self._rl_parameters_all_cheat = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=0.5,
            reward_burnin=10,
            maxq_learning=False,
        )
        self._rl_parameters_all_cheat_maxq = RLParameters(
            gamma=DISCOUNT,
            target_update_rate=1.0,
            reward_burnin=10,
            maxq_learning=True,
        )

    def test_cheat_policy(self):
        state = self._env.reset()
        for _ in range(100):
            state, _, is_terminal, _ = self._env.step('C')
            if is_terminal:
                break
        self.assertTrue(is_terminal)

    def test_pure_q_learning_all_cheat(self):
        q_learning_parameters = DiscreteActionModelParameters(
            actions=self._env.ACTIONS,
            rl=self._rl_parameters_all_cheat_maxq,
            training=TrainingParameters(
                layers=[self._env.width * self._env.height, 1],
                activations=['linear'],
                minibatch_size=32,
                learning_rate=0.05,
                optimizer='SGD',
                lr_policy='fixed'
            )
        )

        trainer = DiscreteActionTrainer(
            self._env.normalization, q_learning_parameters
        )

        predictor = trainer.predictor()

        policy = _build_policy(self._env, predictor, 1)
        initial_state = self._env.reset()
        iteration_result = _collect_samples(
            self._env, policy, 10000, initial_state
        )
        num_iterations = 50
        for _ in range(num_iterations):
            policy = _build_policy(self._env, predictor, 0)

            tdp = self._env.preprocess_samples(
                iteration_result.states,
                iteration_result.actions,
                iteration_result.rewards,
                iteration_result.next_states,
                iteration_result.next_actions,
                iteration_result.is_terminals,
                iteration_result.possible_next_actions,
                None,
            )
            trainer.stream_df(tdp, None)
            initial_state = iteration_result.current_state
        initial_state = self._env.reset()
        iteration_result = _collect_samples(
            self._env, policy, 10000, initial_state
        )
        self.assertTrue(np.all(np.array(iteration_result.actions) == 'C'))

    def test_q_learning_limited(self):
        # TODO: This model oscilliates pretty bad, will investigate in the future.
        target_cheat_percentage = 50
        epsilon = 0.2
        num_iterations = 30
        num_steps = 1024 * 10
        updates_per_iteration = 1

        q_learning_parameters = DiscreteActionModelParameters(
            actions=self._env.ACTIONS,
            rl=self._rl_parameters_maxq,
            training=TrainingParameters(
                layers=[-1, -1],
                activations=['linear'],
                minibatch_size=1024,
                learning_rate=0.05,
                optimizer='ADAM',
            ),
            action_budget=ActionBudget(
                limited_action="C",
                action_limit=target_cheat_percentage,
                quantile_update_rate=0.2,
                quantile_update_frequency=1,
                window_size=1000
            )
        )

        trainer = LimitedActionDiscreteActionTrainer(
            self._env.normalization, q_learning_parameters
        )

        predictor = trainer.predictor()

        policy = _build_policy(self._env, predictor, epsilon)
        initial_state = self._env.reset()
        for iteration in range(num_iterations):
            policy = _build_policy(self._env, predictor, epsilon)
            iteration_result = _collect_samples(
                self._env, policy, num_steps, initial_state
            )
            tdp = self._env.preprocess_samples(
                iteration_result.states,
                iteration_result.actions,
                iteration_result.rewards,
                iteration_result.next_states,
                iteration_result.next_actions,
                iteration_result.is_terminals,
                iteration_result.possible_next_actions,
                None,
            )
            print(
                "iter: {} ({}), ratio: {}, steps to solve: {}, quantile: {}".
                format(
                    iteration, num_steps, iteration_result.cheat_ratio,
                    np.mean(iteration_result.lengths), trainer.quantile_value
                )
            )
            initial_state = iteration_result.current_state
            for _ in range(updates_per_iteration):
                trainer.stream_df(tdp, None)

        state = self._env.reset()
        evaluation_results = _collect_samples(self._env, policy, 10000, state)
        print(
            np.sum(np.array(evaluation_results.lengths) <= 14) /
            len(evaluation_results.lengths)
        )
        optimality_ratio = np.sum(np.array(evaluation_results.lengths) <= 14) / \
            len(evaluation_results.lengths)
        self.assertGreater(optimality_ratio, 0.5)
        accuracy = np.abs(
            evaluation_results.cheat_ratio - target_cheat_percentage / 100
        )
        print(
            "ACCURACY", evaluation_results.cheat_ratio, target_cheat_percentage
        )
        self.assertTrue(
            accuracy < 0.4
        )  # TODO: Would like to get this accuracy up in the future
