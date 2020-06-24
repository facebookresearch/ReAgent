#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List, NamedTuple, Union


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
