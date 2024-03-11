#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import Tuple

import pandas


def generate_discrete_mdp_pandas_df(
    multi_steps: bool, use_seq_num_diff_as_time_diff: bool
) -> Tuple[pandas.DataFrame, str]:
    # Simulate the following MDP:
    # state: 0, action: 7 ('L'), reward: 0,
    # state: 1, action: 8 ('R'), reward: 1,
    # state: 4, action: 9 ('U'), reward: 4,
    # state: 5, action: 10 ('D'), reward: 5,
    # state: 6 (terminal)
    actions = ["L", "R", "U", "D"]
    possible_actions = [["L", "R"], ["R", "U"], ["U", "D"], ["D"]]

    # assume multi_steps=2
    if multi_steps:
        rewards = [[0, 1], [1, 4], [4, 5], [5]]
        metrics = [
            [{"reward": 0}, {"reward": 1}],
            [{"reward": 1}, {"reward": 4}],
            [{"reward": 4}, {"reward": 5}],
            [{"reward": 5}],
        ]
        next_states = [[{1: 1}, {4: 1}], [{4: 1}, {5: 1}], [{5: 1}, {6: 1}], [{6: 1}]]
        next_actions = [["R", "U"], ["U", "D"], ["D", ""], [""]]
        possible_next_actions = [
            [["R", "U"], ["U", "D"]],
            [["U", "D"], ["D"]],
            [["D"], [""]],
            [[""]],
        ]
        # terminals = [[0, 0], [0, 0], [0, 1], [1]]
        time_diffs = [[1, 1], [1, 1], [1, 1], [1]]
    else:
        rewards = [0, 1, 4, 5]
        metrics = [{"reward": 0}, {"reward": 1}, {"reward": 4}, {"reward": 5}]  # noqa
        next_states = [{1: 1}, {4: 1}, {5: 1}, {6: 1}]
        next_actions = ["R", "U", "D", ""]
        possible_next_actions = [["R", "U"], ["U", "D"], ["D"], [""]]
        # terminals = [0, 0, 0, 1]
        if use_seq_num_diff_as_time_diff:
            time_diffs = [1, 1, 1, 1]  # noqa
        else:
            time_diffs = [1, 3, 1, 1]  # noqa

    n = 4
    mdp_ids = ["0", "0", "0", "0"]
    sequence_numbers = [0, 1, 4, 5]
    sequence_number_ordinals = [1, 2, 3, 4]
    states = [{0: 1}, {1: 1}, {4: 1}, {5: 1}]
    action_probabilities = [0.3, 0.4, 0.5, 0.6]

    ds = "2019-07-17"
    df = pandas.DataFrame(
        {
            "mdp_id": mdp_ids,
            "sequence_number": sequence_numbers,
            "sequence_number_ordinal": sequence_number_ordinals,
            "state_features": states,
            "action": actions,
            "action_probability": action_probabilities,
            "reward": rewards,
            "next_state_features": next_states,
            "next_action": next_actions,
            "time_diff": time_diffs,
            "possible_actions": possible_actions,
            "possible_next_actions": possible_next_actions,
            "metrics": metrics,
            "ds": [ds] * n,
        }
    )
    return df, ds


def generate_parametric_mdp_pandas_df(
    multi_steps: bool, use_seq_num_diff_as_time_diff: bool
):
    # Simulate the following MDP:
    # state: 0, action: 7 ('L'), reward: 0,
    # state: 1, action: 8 ('R'), reward: 1,
    # state: 4, action: 9 ('U'), reward: 4,
    # state: 5, action: 10 ('D'), reward: 5,
    # state: 6 (terminal)
    actions = [{7: 1}, {8: 1}, {9: 1}, {10: 1}]
    possible_actions = [
        [{7: 1}, {8: 1}],
        [{8: 1}, {9: 1}],
        [{9: 1}, {10: 1}],
        [{10: 1}],
    ]

    # assume multi_step=2
    if multi_steps:
        rewards = [[0, 1], [1, 4], [4, 5], [5]]
        metrics = [
            [{"reward": 0}, {"reward": 1}],
            [{"reward": 1}, {"reward": 4}],
            [{"reward": 4}, {"reward": 5}],
            [{"reward": 5}],
        ]
        next_states = [[{1: 1}, {4: 1}], [{4: 1}, {5: 1}], [{5: 1}, {6: 1}], [{6: 1}]]
        next_actions = [[{8: 1}, {9: 1}], [{9: 1}, {10: 1}], [{10: 1}, {}], [{}]]
        possible_next_actions = [
            [[{8: 1}, {9: 1}], [{9: 1}, {10: 1}]],
            [[{9: 1}, {10: 1}], [{10: 1}]],
            [[{10: 1}], [{}]],
            [[{}]],
        ]
        # terminals = [[0, 0], [0, 0], [0, 1], [1]]
        time_diffs = [[1, 1], [1, 1], [1, 1], [1]]
    else:
        rewards = [0, 1, 4, 5]
        metrics = [{"reward": 0}, {"reward": 1}, {"reward": 4}, {"reward": 5}]  # noqa
        next_states = [{1: 1}, {4: 1}, {5: 1}, {6: 1}]
        next_actions = [{8: 1}, {9: 1}, {10: 1}, {}]
        possible_next_actions = [[{8: 1}, {9: 1}], [{9: 1}, {10: 1}], [{10: 1}], [{}]]
        # terminals = [0, 0, 0, 1]
        if use_seq_num_diff_as_time_diff:
            time_diffs = [1, 1, 1, 1]  # noqa
        else:
            time_diffs = [1, 3, 1, 1]  # noqa

    n = 4
    mdp_ids = ["0", "0", "0", "0"]
    sequence_numbers = [0, 1, 4, 5]
    sequence_number_ordinals = [1, 2, 3, 4]
    states = [{0: 1}, {1: 1}, {4: 1}, {5: 1}]
    action_probabilities = [0.3, 0.4, 0.5, 0.6]

    ds = "2019-07-17"
    df = pandas.DataFrame(
        {
            "mdp_id": mdp_ids,
            "sequence_number": sequence_numbers,
            "sequence_number_ordinal": sequence_number_ordinals,
            "state_features": states,
            "action": actions,
            "action_probability": action_probabilities,
            "reward": rewards,
            "next_state_features": next_states,
            "next_action": next_actions,
            "time_diff": time_diffs,
            "possible_actions": possible_actions,
            "possible_next_actions": possible_next_actions,
            "metrics": metrics,
            "ds": [ds] * n,
        }
    )
    return df, ds
