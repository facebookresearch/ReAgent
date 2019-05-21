#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import collections
import random
from functools import partial
from typing import Deque, Dict, List, NamedTuple, Optional, Union


FEATURES = Dict[int, float]
ACTION = Union[str, FEATURES]


class Samples(NamedTuple):
    mdp_ids: List[str]
    sequence_numbers: List[int]
    sequence_number_ordinals: List[int]
    states: List[FEATURES]
    actions: List[ACTION]
    action_probabilities: List[float]
    rewards: List[float]
    possible_actions: List[List[ACTION]]
    next_states: List[FEATURES]
    next_actions: List[ACTION]
    terminals: List[bool]
    possible_next_actions: List[List[ACTION]]


class MultiStepSamples(NamedTuple):
    mdp_ids: List[str]
    sequence_numbers: List[int]
    sequence_number_ordinals: List[int]
    states: List[FEATURES]
    actions: List[ACTION]
    action_probabilities: List[float]
    rewards: List[List[float]]
    possible_actions: List[List[ACTION]]
    next_states: List[List[FEATURES]]
    next_actions: List[List[ACTION]]
    terminals: List[List[bool]]
    possible_next_actions: List[List[List[ACTION]]]

    def to_single_step(self) -> Samples:
        return Samples(
            mdp_ids=self.mdp_ids,
            sequence_numbers=self.sequence_numbers,
            sequence_number_ordinals=self.sequence_number_ordinals,
            states=self.states,
            actions=self.actions,
            action_probabilities=self.action_probabilities,
            rewards=[r[0] for r in self.rewards],
            possible_actions=self.possible_actions,
            next_states=[ns[0] for ns in self.next_states],
            next_actions=[na[0] for na in self.next_actions],
            terminals=[t[0] for t in self.terminals],
            possible_next_actions=[pna[0] for pna in self.possible_next_actions],
        )


def shuffle_samples(samples):
    merged = list(zip(*[getattr(samples, f) for f in samples._fields]))
    random.shuffle(merged)
    return type(samples)(**dict(zip(samples._fields, zip(*merged))))


class Environment:
    def reset(self):
        """ Reset the environment and return the initial state """
        pass

    def step(self, action):
        """
        Proceed one step ahead using the action.
        Return next state, reward, terminal, and info
        """
        return None, None, None, None

    def _process_state(self, state):
        """
        Transform the state to the format that can be uploaded to Hive
        """
        pass

    def sample_policy(self, state, use_continuous_action, epsilon):
        """
        Sample an action following epsilon-greedy
        Return the raw action which can be fed into env.step(), the processed
            action which can be uploaded to Hive, and action probability
        """
        return None, None, None

    def action_to_features(self, action) -> FEATURES:
        """
        Transform an action into a feature vector (as a dictionary)
        Call this function when discrete actions need to be transformed into
        continuous formats
        """
        raise NotImplementedError

    def possible_actions(
        self,
        state,
        terminal=False,
        ignore_terminal=False,
        use_continuous_action: bool = False,
        **kwargs,
    ) -> List[ACTION]:
        """
        Get possible actions at the current state. If ignore_terminal is False,
        then this function always returns an empty list at a terminal state.
        """
        pass

    @staticmethod
    def set_if_in_range(index, limit, container, value):
        if index >= limit:
            return
        container[index] = value

    def generate_random_samples(
        self,
        num_transitions: int,
        use_continuous_action: bool,
        epsilon: float = 1.0,
        multi_steps: Optional[int] = None,
        max_step: Optional[int] = None,
    ) -> Union[Samples, MultiStepSamples]:
        """ Generate samples:
            [
             s_t,
             (a_t, a_{t+1}, ..., a_{t+steps}),
             (r_t, r_{t+1}, ..., r_{t+steps}),
             (s_{t+1}, s_{t+2}, ..., s_{t+steps+1})
            ]

        :param num_transitions: How many transitions to collect
        :param use_continuous_action: True if a discrete action needs to be
            represented as a vector using a dictionary; otherwise the action is
            represented as string.
        :param epsilon: (1-epsilon) determines the chance of taking optimal actions.
            Only valid when the environment (e.g., gridworld) records optimal actions.
        :param multi_steps: An integer decides how many steps of transitions
            contained in each sample. Only used if you want to train multi-step RL.
        :param max_step: An episode terminates after max_step number of steps
        """
        return_single_step_samples = False
        if multi_steps is None:
            return_single_step_samples = True
            multi_steps = 1

        states: List[FEATURES] = [{} for _ in range(num_transitions)]
        action_probabilities: List[float] = [0.0] * num_transitions
        rewards: List[List[float]] = [[] for _ in range(num_transitions)]
        next_states: List[List[FEATURES]] = [[{}] for _ in range(num_transitions)]
        terminals: List[List[bool]] = [[] for _ in range(num_transitions)]
        mdp_ids = [""] * num_transitions
        sequence_numbers = [0] * num_transitions
        possible_actions: List[List[ACTION]] = [[] for _ in range(num_transitions)]
        possible_next_actions: List[List[List[ACTION]]] = [
            [[]] for _ in range(num_transitions)
        ]
        next_actions: List[List[ACTION]] = [[] for _ in range(num_transitions)]
        actions: List[ACTION] = []
        if use_continuous_action:
            actions = [{} for _ in range(num_transitions)]
        else:
            actions = [""] * num_transitions

        state = None
        terminal = True
        raw_action = None
        processed_action = None
        next_raw_action = None
        next_processed_action = None
        next_action_probability = 1.0
        transition = 0
        mdp_id = -1
        sequence_number = 0

        state_deque: Deque[FEATURES] = collections.deque(maxlen=multi_steps)
        action_deque: Deque[ACTION] = collections.deque(maxlen=multi_steps)
        action_probability_deque: Deque[float] = collections.deque(maxlen=multi_steps)
        reward_deque: Deque[float] = collections.deque(maxlen=multi_steps)
        next_state_deque: Deque[FEATURES] = collections.deque(maxlen=multi_steps)
        next_action_deque: Deque[ACTION] = collections.deque(maxlen=multi_steps)
        terminal_deque: Deque[bool] = collections.deque(maxlen=multi_steps)
        sequence_number_deque: Deque[int] = collections.deque(maxlen=multi_steps)
        possible_action_deque: Deque[List[ACTION]] = collections.deque(
            maxlen=multi_steps
        )
        possible_next_action_deque: Deque[List[ACTION]] = collections.deque(
            maxlen=multi_steps
        )

        # We run until we finish the episode that completes N transitions, but
        # we may have to go beyond N to reach the end of that episode
        while not terminal or transition < num_transitions:
            if terminal:
                state = self.reset()
                terminal = False
                mdp_id += 1
                sequence_number = 0
                state_deque.clear()
                action_deque.clear()
                action_probability_deque.clear()
                reward_deque.clear()
                next_state_deque.clear()
                next_action_deque.clear()
                terminal_deque.clear()
                sequence_number_deque.clear()
                possible_action_deque.clear()
                possible_next_action_deque.clear()
                raw_action, processed_action, action_probability = self.sample_policy(
                    state, use_continuous_action, epsilon
                )
            else:
                raw_action = next_raw_action
                processed_action = next_processed_action
                action_probability = next_action_probability
                sequence_number += 1

            possible_action = self.possible_actions(
                state,
                terminal=terminal,
                ignore_terminal=False,
                use_continuous_action=use_continuous_action,
            )
            next_state, reward, terminal, _ = self.step(raw_action)
            if max_step is not None and sequence_number >= max_step:
                terminal = True
            next_raw_action, next_processed_action, next_action_probability = self.sample_policy(
                next_state, use_continuous_action, epsilon
            )
            possible_next_action = self.possible_actions(
                next_state,
                terminal=terminal,
                ignore_terminal=False,
                use_continuous_action=use_continuous_action,
            )

            state_deque.append(self._process_state(state))
            action_deque.append(processed_action)
            action_probability_deque.append(action_probability)
            reward_deque.append(reward)
            terminal_deque.append(terminal)
            sequence_number_deque.append(sequence_number)
            possible_action_deque.append(possible_action)
            possible_next_action_deque.append(possible_next_action)

            next_processed_state: FEATURES = self._process_state(next_state)
            next_state_deque.append(next_processed_state)

            # Format terminals in same way we ask clients to log terminals (in RL dex)
            # i.e., setting next action empty if the episode terminates
            if terminal:
                # We need to keep next state even at the terminal state
                # first, fblearner/flow/projects/rl/core/data_fetcher.py decides
                # terminal signals by looking at next action, not next state
                # second, next state will be used for world model building
                if type(next_processed_action) is str:
                    next_processed_action = ""
                else:
                    next_processed_action = {}
            next_action_deque.append(next_processed_action)

            # We want exactly N data points, but we need to wait until the
            # episode is over so we can get the episode values. `set_if_in_range`
            # will set episode values if they are in the range [0,N) and ignore
            # otherwise.
            if not terminal and len(state_deque) == multi_steps:
                set_if_in_range = partial(
                    self.set_if_in_range, transition, num_transitions
                )
                set_if_in_range(states, state_deque[0])
                set_if_in_range(actions, action_deque[0])
                set_if_in_range(action_probabilities, action_probability_deque[0])
                set_if_in_range(rewards, list(reward_deque))
                set_if_in_range(next_states, list(next_state_deque))
                set_if_in_range(next_actions, list(next_action_deque))
                set_if_in_range(terminals, list(terminal_deque))
                set_if_in_range(mdp_ids, str(mdp_id))
                set_if_in_range(sequence_numbers, sequence_number_deque[0])
                set_if_in_range(possible_actions, possible_action_deque[0])
                set_if_in_range(possible_next_actions, list(possible_next_action_deque))
                transition += 1
            # collect samples at the end of the episode. The steps between state
            # and next_state can be less than or equal to `multi_steps`
            if terminal:
                for _ in range(len(state_deque)):
                    set_if_in_range = partial(
                        self.set_if_in_range, transition, num_transitions
                    )
                    set_if_in_range(states, state_deque.popleft())
                    set_if_in_range(actions, action_deque.popleft())
                    set_if_in_range(
                        action_probabilities, action_probability_deque.popleft()
                    )
                    set_if_in_range(rewards, list(reward_deque))
                    set_if_in_range(next_states, list(next_state_deque))
                    set_if_in_range(next_actions, list(next_action_deque))
                    set_if_in_range(terminals, list(terminal_deque))
                    set_if_in_range(mdp_ids, str(mdp_id))
                    set_if_in_range(sequence_numbers, sequence_number_deque.popleft())
                    set_if_in_range(possible_actions, possible_action_deque.popleft())
                    set_if_in_range(
                        possible_next_actions, list(possible_next_action_deque)
                    )
                    reward_deque.popleft()
                    next_state_deque.popleft()
                    next_action_deque.popleft()
                    terminal_deque.popleft()
                    possible_next_action_deque.popleft()
                    transition += 1

            state = next_state

        samples = MultiStepSamples(
            mdp_ids=mdp_ids,
            sequence_numbers=sequence_numbers,
            sequence_number_ordinals=sequence_numbers,
            states=states,
            actions=actions,
            action_probabilities=action_probabilities,
            rewards=rewards,
            possible_actions=possible_actions,
            next_states=next_states,
            next_actions=next_actions,
            terminals=terminals,
            possible_next_actions=possible_next_actions,
        )
        if return_single_step_samples:
            return samples.to_single_step()
        return samples
