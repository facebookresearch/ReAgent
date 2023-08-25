#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
from abc import abstractmethod
from typing import Optional

from reagent.ope.estimators.sequential_estimators import (
    Mdp,
    Model,
    RLPolicy,
    State,
    StateReward,
    Transition,
)


class Environment(Model):
    """
    Environment for RL
    """

    def __init__(self, max_horizon: int = -1) -> None:
        self._current_state: Optional[State] = None
        self._steps_taken: int = 0
        self._max_horizon = max_horizon

    @abstractmethod
    def reset(self, state: Optional[State] = None):
        self.current_state = state
        self._steps_taken = 0

    @abstractmethod
    def close(self):
        pass

    def step(self, policy: RLPolicy):
        a_dist = policy(self.current_state)
        a = a_dist.sample()[0]
        s_dist = self(self.current_state, a)
        srs = []
        probs = []
        for s, rp in s_dist.items():
            srs.append(StateReward(s, rp.reward))
            probs.append(rp.prob)
        sr = random.choices(srs, weights=probs)[0]

        last_state = self.current_state
        noop = sr.state == self.current_state
        if not noop:
            self.current_state = sr.state
        self._steps_taken += 1

        status = Transition.Status.NORMAL
        if 0 < self._max_horizon <= self._steps_taken or self.current_state.is_terminal:
            status = Transition.Status.TERMINATED
        elif noop:
            status = Transition.Status.NOOP
        return Transition(
            last_state=last_state,
            action=a,
            action_prob=a_dist[a],
            state=self.current_state,
            reward=sr.reward,
            status=status,
        )

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @property
    @abstractmethod
    def states(self):
        pass

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state: Optional[State]):
        self._current_state = state


class PolicyLogGenerator:
    def __init__(self, env: Environment, policy: RLPolicy) -> None:
        self._env = env
        self._policy = policy

    def generate_log(self, init_state: State, max_horizon: int = -1) -> Mdp:
        transition = Transition(state=self._env.reset(state=init_state))
        mpd = []
        while transition.status != Transition.Status.TERMINATED:
            if max_horizon > 0 and len(mpd) > max_horizon:
                break
            transition = self._env.step(self._policy)
            mpd.append(transition)
        return mpd
