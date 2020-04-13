#!/usr/bin/env python3

import logging
import random
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from reagent.ope.estimators.sequential_estimators import (
    DMEstimator,
    DREstimator,
    EpsilonGreedyRLPolicy,
    IPSEstimator,
    MAGICEstimator,
    RandomRLPolicy,
    RewardProbability,
    RLEstimatorInput,
    State,
    StateDistribution,
    StateReward,
    ValueFunction,
)
from reagent.ope.estimators.types import Action, ActionSpace
from reagent.ope.test.envs import Environment, PolicyLogGenerator
from reagent.ope.trainers.rl_tabular_trainers import (
    DPTrainer,
    DPValueFunction,
    TabularPolicy,
)


class GridWorld(Environment):
    def __init__(
        self,
        size: Tuple[int, int],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_horizon: int = -1,
        walls: Iterable[Tuple[int, int]] = (),
    ):
        super().__init__(max_horizon)
        self.size = size
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        self.reset()

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[str]], max_horizon: int = -1):
        size = (len(grid), len(grid[0]))
        start = ()
        goal = ()
        walls = []
        for x, r in enumerate(grid):
            for y, c in enumerate(r):
                g = c.lower()
                if g == "s":
                    start = (x, y)
                elif g == "g":
                    goal = (x, y)
                elif g == "w":
                    walls += ((x, y),)
        return cls(size, start, goal, max_horizon, walls)

    def reset(self, state: Optional[State] = None):
        super().reset(state)
        if self._current_state is None:
            self._current_state = State(self.start)
        return self._current_state

    def close(self):
        pass

    def _validate(self, pos: Tuple[int, int]) -> bool:
        return (
            0 <= pos[0] < self.size[0]
            and 0 <= pos[1] < self.size[1]
            and pos not in self.walls
        )

    def _transit(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], float, bool]:
        if not self._validate(to_pos):
            return from_pos, 0.0, False
        elif to_pos == self.goal:
            return to_pos, 1.0, True
        else:
            return to_pos, 0.0, False

    def _next_state_reward(self, state: State, action: Action) -> StateReward:
        x, y = state.value
        if state.value in self.walls or state.value == self.goal:
            return StateReward(State((x, y), state.is_terminal), 0.0)
        if action.value == 0:
            to_pos, reward, is_end = self._transit((x, y), (x + 1, y))
        elif action.value == 1:
            to_pos, reward, is_end = self._transit((x, y), (x, y + 1))
        elif action.value == 2:
            to_pos, reward, is_end = self._transit((x, y), (x - 1, y))
        else:
            to_pos, reward, is_end = self._transit((x, y), (x, y - 1))
        return StateReward(State(to_pos, is_end), reward)

    def next_state_reward_dist(self, state: State, action: Action) -> StateDistribution:
        sr = self._next_state_reward(state, action)
        return {sr.state: RewardProbability(sr.reward, 1.0)}

    @property
    def observation_space(self):
        return (2,), ((0, self.size[0]), (0, self.size[1]))

    @property
    def states(self):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                state = (x, y)
                if state != self.goal and state not in self.walls:
                    yield State((x, y))

    def __repr__(self):
        dump = ""
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                pos = (x, y)
                if pos == self.start:
                    dump += "\u2b55"
                elif pos == self.goal:
                    dump += "\u2b50"
                elif pos in self.walls:
                    dump += "\u2b1b"
                else:
                    dump += "\u2b1c"
            dump += "\n"
        return dump

    def dump_state_values(self, state_values) -> str:
        dump = ""
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                pos = State((x, y))
                value = 0.0
                if pos in state_values:
                    value = state_values[pos]
                dump += "{:6.3}".format(value)
            dump += "\n"
        return dump

    def dump_value_func(self, valfunc: ValueFunction) -> str:
        dump = ""
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                dump += "{:6.3}".format(valfunc(State((x, y))))
            dump += "\n"
        return dump

    def dump_policy(self, policy) -> str:
        dump = ""
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                pos = (x, y)
                if pos == self.start:
                    dump += "\u2b28"
                elif pos == self.goal:
                    dump += "\u2b27"
                elif pos in self.walls:
                    dump += "\u2588"
                else:
                    action = policy(State(pos)).greedy()
                    if action.value == 0:
                        dump += "\u21e9"
                    elif action.value == 1:
                        dump += "\u21e8"
                    elif action.value == 2:
                        dump += "\u21e7"
                    else:
                        dump += "\u21e6"
            dump += "\n"
        return dump


class ThomasGridWorld(GridWorld):
    """
    GridWorld set up in https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf
    """

    def __init__(self):
        super().__init__((4, 4), (0, 0), (3, 3), 100)

    def _transit(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], float, bool]:
        if not self._validate(to_pos):
            return from_pos, 0.0, False
        elif to_pos == (1, 2):
            return to_pos, -10.0, False
        elif to_pos == (1, 3):
            return to_pos, 1.0, False
        elif to_pos == self.goal:
            return to_pos, 10.0, True
        else:
            return to_pos, -1.0, False


class NoiseGridWorldModel(Environment):
    def __init__(
        self,
        gridworld: GridWorld,
        action_space: ActionSpace,
        epsilon: float = 0.1,
        max_horizon: int = -1,
    ):
        super().__init__(max_horizon)
        self._gridworld = gridworld
        self.action_space = action_space
        self.epsilon = epsilon
        self.noise_prob = epsilon / (len(action_space) - 1)

    def reset(self, state: Optional[State] = None):
        self._gridworld.reset(state)
        self._steps_taken = 0
        return self._gridworld.current_state

    def close(self):
        pass

    def next_state_reward_dist(self, state: State, action: Action) -> StateDistribution:
        probs = [self.noise_prob] * len(self.action_space)
        probs[action.value] = 1 - self.epsilon
        states = {}
        for a in self.action_space:
            sr = self._gridworld._next_state_reward(state, a)
            if sr.state in states:
                rp = states[sr.state]
                states[sr.state] = RewardProbability(
                    rp.reward + sr.reward, rp.prob + probs[a.value]
                )
            else:
                states[sr.state] = RewardProbability(sr.reward, probs[a.value])
        return states

    @property
    def observation_space(self):
        return self._gridworld.observation_space

    @property
    def states(self):
        return self._gridworld.states

    @property
    def current_state(self):
        return self._gridworld.current_state

    @current_state.setter
    def current_state(self, state: Optional[None]):
        self._gridworld._current_state = state


GAMMA = 0.9

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    random.seed(1234)
    np.random.seed(1234)
    torch.random.manual_seed(1234)

    device = torch.device("cuda") if torch.cuda.is_available() else None
    print(f"device - {device}")

    gridworld = GridWorld.from_grid(
        [
            ["s", "0", "0", "0", "0"],
            ["0", "0", "0", "W", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "W", "0", "0", "0"],
            ["0", "0", "0", "0", "g"],
        ],
        # [
        #     ["s", "0", "0", "0"],
        #     ["0", "0", "0", "0"],
        #     ["0", "0", "0", "0"],
        #     ["0", "0", "0", "g"],
        # ],
        max_horizon=1000,
    )
    # gridworld = ThomasGridWorld()
    logging.info(f"GridWorld:\n{gridworld}")

    action_space = ActionSpace(4)
    opt_policy = TabularPolicy(action_space)
    trainer = DPTrainer(gridworld, opt_policy)
    value_func = trainer.train(gamma=GAMMA)

    logging.info(f"Opt Policy:\n{gridworld.dump_policy(opt_policy)}")
    logging.info(f"Opt state values:\n{gridworld.dump_value_func(value_func)}")

    behavivor_policy = RandomRLPolicy(action_space)
    target_policy = EpsilonGreedyRLPolicy(opt_policy, 0.3)
    model = NoiseGridWorldModel(gridworld, action_space, epsilon=0.3, max_horizon=1000)
    value_func = DPValueFunction(target_policy, model, GAMMA)
    ground_truth = DPValueFunction(target_policy, gridworld, GAMMA)

    logging.info(
        f"Target Policy ground truth values:\n"
        f"{gridworld.dump_value_func(ground_truth)}"
    )

    log = {}
    log_generator = PolicyLogGenerator(gridworld, behavivor_policy)
    num_episodes = 200
    for state in gridworld.states:
        mdps = []
        for _ in range(num_episodes):
            mdps.append(log_generator.generate_log(state))
        log[state] = mdps
        logging.info(f"Generated {len(mdps)} logs for {state}")

    estimator_input = RLEstimatorInput(
        gamma=GAMMA,
        log=log,
        target_policy=target_policy,
        value_function=value_func,
        ground_truth=ground_truth,
    )

    DMEstimator(device=device).evaluate(estimator_input)

    IPSEstimator(weight_clamper=None, weighted=False, device=device).evaluate(
        estimator_input
    )
    IPSEstimator(weight_clamper=None, weighted=True, device=device).evaluate(
        estimator_input
    )

    DREstimator(weight_clamper=None, weighted=False, device=device).evaluate(
        estimator_input
    )
    DREstimator(weight_clamper=None, weighted=True, device=device).evaluate(
        estimator_input
    )

    MAGICEstimator(device=device).evaluate(
        estimator_input, num_resamples=10, loss_threhold=0.0000001, lr=0.00001
    )
