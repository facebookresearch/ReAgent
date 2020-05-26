#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


logger = logging.getLogger(__name__)

DEFAULT_DS = "2019-01-01"


def _dense_to_sparse(dense: np.ndarray) -> List[Dict[str, float]]:
    """ Convert dense array to sparse representation """
    assert len(dense.shape) == 2, f"dense shape is {dense.shape}"
    # pyre-fixme[7]: Expected `List[Dict[str, float]]` but got `List[Dict[int,
    #  typing.Any]]`.
    return [{i: v.item() for i, v in enumerate(elem)} for elem in dense]


def replay_buffer_to_pre_timeline_df(
    is_discrete_action: bool, replay_buffer: ReplayBuffer
) -> pd.DataFrame:
    """ Format needed for uploading dataset to Hive, and then run timeline. """
    n = replay_buffer.size
    batch = replay_buffer.sample_transition_batch_tensor(batch_size=n)

    # actions is inconsistent between models, so let's infer them.
    possible_actions_mask = getattr(batch, "possible_actions_mask", None)
    possible_actions = getattr(batch, "possible_actions", None)

    terminal = batch.terminal.squeeze(1).tolist()
    assert len(batch.action.shape) == 2
    if is_discrete_action:
        assert (
            batch.action.shape[1] == 1
        ), f"discrete action batch with shape {batch.action.shape}"
        # Discrete action space, should be str
        action = [str(a.item()) for a in batch.action]
        # assuming we've explored the whole action space
        unique_actions = np.unique(batch.action)
        possible_actions_mask = [
            [1 for _ in range(len(unique_actions))] if not elem_terminal else []
            for elem_terminal in terminal
        ]
        possible_actions = [
            [str(a) for a in unique_actions] if not elem_terminal else []
            for elem_terminal in terminal
        ]
    else:
        # Box (parametric) action space, should be map<str, double>
        action = _dense_to_sparse(batch.action)

        # TODO: handle possible actions/mask here

    sequence_number = batch.sequence_number.squeeze(1).tolist()
    action_probability = np.exp(batch.log_prob.squeeze(1)).tolist()
    reward = batch.reward.squeeze(1).tolist()
    rows = {
        "ds": [DEFAULT_DS for _ in range(n)],
        "state_features": _dense_to_sparse(batch.state),
        "action": action,
        "mdp_id": batch.mdp_id.tolist(),
        "sequence_number": sequence_number,
        "action_probability": action_probability,
        "reward": reward,
        "metrics": [{"reward": r} for r in reward],
    }

    if possible_actions_mask is not None:
        rows["possible_actions_mask"] = possible_actions_mask

    if possible_actions is not None:
        rows["possible_actions"] = possible_actions

    return pd.DataFrame.from_dict(rows)
