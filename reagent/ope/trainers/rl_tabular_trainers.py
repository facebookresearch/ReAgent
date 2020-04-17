#!/usr/bin/env python3

import pickle
from functools import reduce
from typing import Mapping, Sequence

import torch
from reagent.ope.estimators.estimator import (
    Action,
    ActionDistribution,
    ActionSpace,
    Model,
    RLPolicy,
    State,
    ValueFunction,
)
from reagent.ope.test.envs import Environment, PolicyLogGenerator


class TabularPolicy(RLPolicy):
    def __init__(self, action_space: ActionSpace, epsilon: float = 0.0, device=None):
        super().__init__(action_space, device)
        self._epsilon = epsilon
        as_size = len(action_space)
        self._exploitation_prob = 1.0 - epsilon
        self._exploration_prob = epsilon / len(action_space)
        self._uniform_probs = as_size * [1.0 / as_size]
        self._state_space = {}

    def update(self, state: State, actions: Sequence[float]) -> float:
        old_dist = self._uniform_probs
        if state in self._state_space:
            old_dist = self._state_space[state]
        self._state_space[state] = actions
        return (
            reduce(
                lambda a, b: a + b,
                map(lambda a: (a[0] - a[1]) ** 2, zip(old_dist, actions)),
            )
            ** 0.5
        )

    def action_dist(self, state: State) -> ActionDistribution:
        if state in self._state_space:
            actions = self._state_space[state]
            probs = list(
                map(
                    lambda p: p * self._exploitation_prob + self._exploration_prob,
                    actions,
                )
            )
        else:
            probs = self._uniform_probs
        return self._action_space.distribution(probs)

    def save(self, path) -> bool:
        try:
            with open(path, "wb") as f:
                pickle.dump((self._action_space, self._epsilon, self._state_space), f)
        except Exception:
            return False
        else:
            return True

    def load(self, path) -> bool:
        try:
            with open(path, "rb") as f:
                self._action_space, self._epsilon, self._state_space = pickle.load(f)
        except Exception:
            return False
        else:
            return True


class TabularValueFunction(ValueFunction):
    def __init__(self, policy: RLPolicy, model: Model, gamma=0.99):
        self._policy = policy
        self._model = model
        self._gamma = gamma
        self._state_values = {}

    def _state_value(self, state: State) -> float:
        return (
            0.0
            if (state is None or state.is_terminal)
            else (0.0 if state not in self._state_values else self._state_values[state])
        )

    def state_action_value(self, state: State, action: Action) -> float:
        value = 0.0
        sr_dist = self._model.next_state_reward_dist(state, action)
        for _, rp in sr_dist.items():
            value += rp.prob * (rp.reward + self._gamma * self.state_value(state))
        return value

    def state_value(self, state: State) -> float:
        pass

    def reset(self, clear_state_values: bool = False):
        pass


class DPValueFunction(TabularValueFunction):
    def __init__(
        self,
        policy: RLPolicy,
        env: Environment,
        gamma: float = 0.99,
        threshold: float = 0.0001,
    ):
        super().__init__(policy, env, gamma)
        self._env = env
        self._threshold = threshold
        self._evaluated = False

    def state_value(self, state: State, horizon: int = -1) -> float:
        if not self._evaluated:
            self._evaluate()
        return self._state_value(state)

    def reset(self, clear_state_values: bool = False):
        self._evaluated = False
        if clear_state_values:
            self._state_values.clear()

    def _evaluate(self):
        delta = float("inf")
        while delta >= self._threshold:
            delta = 0.0
            for state in self._env.states:
                old_value = self._state_value(state)
                new_value = 0.0
                a_dist = self._policy(state)
                for a, ap in a_dist:
                    s_dist = self._model(state, a)
                    a_value = 0.0
                    for s, rp in s_dist.items():
                        a_value += rp.prob * (
                            rp.reward + self._gamma * self._state_value(s)
                        )
                    new_value += ap * a_value
                delta = max(delta, abs(old_value - new_value))
                self._state_values[state] = new_value
        self._evaluated = True


class DPTrainer(object):
    def __init__(self, env: Environment, policy: TabularPolicy):
        self._env = env
        self._policy = policy

    @staticmethod
    def _state_value(state: State, state_values: Mapping[State, float]) -> float:
        return 0.0 if state not in state_values else state_values[state]

    def train(self, gamma: float = 0.9, threshold: float = 0.0001):
        stable = False
        valfunc = DPValueFunction(self._policy, self._env, gamma, threshold)
        while not stable:
            stable = True
            for state in self._env.states:
                new_actions = []
                max_value = float("-inf")
                for action in self._policy.action_space:
                    s_dist = self._env(state, action)
                    value = 0.0
                    for s, rp in s_dist.items():
                        value += rp.prob * rp.reward
                        if s is not None:
                            value += rp.prob * gamma * valfunc(s)
                    if value > max_value:
                        max_value = value
                        new_actions = [action]
                    elif value == max_value:
                        new_actions.append(action)
                prob = 1.0 / len(new_actions)
                actions = [0.0] * len(self._policy.action_space)
                for a in new_actions:
                    actions[a.value] = prob
                if self._policy.update(state, actions) >= 1.0e-6:
                    stable = False
            valfunc.reset()
        return valfunc


class MonteCarloValueFunction(TabularValueFunction):
    def __init__(
        self,
        policy: RLPolicy,
        env: Environment,
        gamma: float = 0.99,
        first_visit: bool = True,
        count_threshold: int = 100,
        max_iteration: int = 200,
    ):
        super().__init__(policy, env, gamma)
        self._env = env
        self._first_visit = first_visit
        self._count_threshold = count_threshold
        self._max_iteration = max_iteration
        self._log_generator = PolicyLogGenerator(env, policy)
        self._state_counts = {}

    def _state_value(self, state: State):
        i = 0
        state_count = self._state_counts[state] if state in self._state_counts else 0
        while state_count < self._count_threshold and i < self._max_iteration:
            i += 1
            mdp = self._log_generator.generate_log(state)
            if self._first_visit:
                state_counts = {}
                for t in mdp:
                    if t.last_state is None:
                        continue
                    if t.last_state in state_counts:
                        state_counts[t.last_state] += 1
                    else:
                        state_counts[t.last_state] = 1
                g = 0
                for t in reversed(mdp):
                    if t.last_state is None:
                        continue
                    g = self._gamma * g + t.reward
                    counts = state_counts[t.last_state]
                    if counts > 1:
                        self._update_state_value(t.last_state, g)
                    counts -= 1
                    if counts == 0:
                        del state_counts[t.last_state]
                    else:
                        state_counts[t.last_state] = counts
            else:
                g = 0
                for t in reversed(mdp):
                    if t.last_state is None:
                        continue
                    g = self._gamma * g + t.reward
                    self._update_state_value(t.last_state, g)
            state_count = (
                self._state_counts[state] if state in self._state_counts else 0
            )
        return super()._state_value(state)

    def _update_state_value(self, state: State, g: float):
        sv = super()._state_value(state)
        sc = self._state_counts[state] if state in self._state_counts else 0
        sc += 1
        sv = sv + (g - sv) / sc
        self._state_values[state] = sv
        self._state_counts[state] = sc

    def state_value(self, state: State) -> float:
        return self._state_value(state)

    def reset(self, clear_state_values: bool = False):
        if clear_state_values:
            self._state_values.clear()
            self._state_counts.clear()


class MonteCarloTrainer(object):
    def __init__(self, env: Environment, policy: TabularPolicy):
        self._env = env
        self._policy = policy
        self._log_generator = PolicyLogGenerator(env, policy)

    def train(
        self,
        iterations: int,
        gamma: float = 0.9,
        first_visit: bool = True,
        update_interval: int = 20,
    ):
        i = 0
        value_counts = {}
        while i < iterations:
            i += 1
            for state in self._env.states:
                mdp = self._log_generator.generate_log(state)
                if first_visit:
                    vcounts = {}
                    for t in mdp:
                        if t.last_state is None or t.action is None:
                            continue
                        key = (t.last_state, t.action)
                        if key in vcounts:
                            vcounts[key] += 1
                        else:
                            vcounts[key] = 1
                    g = 0
                    for t in reversed(mdp):
                        if t.last_state is None or t.action is None:
                            continue
                        g = gamma * g + t.reward
                        key = (t.last_state, t.action)
                        vc = vcounts[key]
                        if vc > 1:
                            self._update_state_value(
                                value_counts, t.last_state, t.action, g
                            )
                        vc -= 1
                        if vc == 0:
                            del vcounts[key]
                        else:
                            vcounts[key] = vc
                else:
                    g = 0
                    for t in reversed(mdp):
                        if t.last_state is None or t.action is None:
                            continue
                        g = gamma * g + t.reward
                        self._update_state_value(
                            value_counts, t.last_state, t.action, g
                        )
            if i % update_interval == 0 and self._update_policy(value_counts):
                break

    def _update_state_value(self, value_counts, state, action, g: float):
        key = (state, action)
        sv, sc = value_counts[key] if key in value_counts else (0.0, 0)
        sc += 1
        sv = sv + (g - sv) / sc
        value_counts[key] = (sv, sc)

    def _update_policy(self, value_counts) -> bool:
        stable = True
        for state in self._env.states:
            probs = []
            for a in self._policy.action_space:
                key = (state, a)
                if key not in value_counts:
                    probs.append(0.0)
                else:
                    v, c = value_counts[key]
                    probs.append(v * c)
            probs = torch.nn.functional.softmax(torch.tensor(probs), dim=0).tolist()
            if self._policy.update(state, probs) >= 1.0e-6:
                stable = False
        return stable
