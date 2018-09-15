#!/usr/bin/env python3

from typing import Optional

import numpy as np
import torch


class TrainingDataPage(object):
    __slots__ = [
        "mdp_ids",
        "sequence_numbers",
        "states",
        "actions",
        "propensities",
        "rewards",
        "possible_actions",
        "possible_actions_lengths",
        "state_pas_concat",
        "next_states",
        "next_actions",
        "possible_next_actions",
        "possible_next_actions_lengths",
        "episode_values",
        "not_terminals",
        "time_diffs",
        "possible_next_actions_state_concat",
    ]

    def __init__(
        self,
        mdp_ids: Optional[np.ndarray] = None,
        sequence_numbers: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        propensities: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        possible_actions: Optional[torch.Tensor] = None,
        possible_actions_lengths: Optional[torch.Tensor] = None,
        state_pas_concat: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
        next_actions: Optional[torch.Tensor] = None,
        possible_next_actions: Optional[torch.Tensor] = None,
        episode_values: Optional[torch.Tensor] = None,
        not_terminals: Optional[torch.Tensor] = None,
        time_diffs: Optional[torch.Tensor] = None,
        possible_next_actions_lengths: Optional[torch.Tensor] = None,
        possible_next_actions_state_concat: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Creates a TrainingDataPage object.

        In the case where `not_terminals` can be determined by next_actions or
        possible_next_actions, feel free to omit it.
        """
        self.mdp_ids = mdp_ids
        self.sequence_numbers = sequence_numbers
        self.states = states
        self.actions = actions
        self.propensities = propensities
        self.rewards = rewards
        self.possible_actions = possible_actions
        self.possible_actions_lengths = possible_actions_lengths
        self.state_pas_concat = state_pas_concat
        self.next_states = next_states
        self.next_actions = next_actions
        self.possible_next_actions = possible_next_actions
        self.episode_values = episode_values
        self.not_terminals = not_terminals
        self.time_diffs = time_diffs
        self.possible_next_actions_lengths = possible_next_actions_lengths
        self.possible_next_actions_state_concat = possible_next_actions_state_concat

    def size(self) -> int:
        if self.states:
            return len(self.states)
        raise Exception("Cannot get size of TrainingDataPage missing states.")

    def set_type(self, dtype):
        # TODO: Clean this up in a future diff.  Figure out which should be long/float
        for x in TrainingDataPage.__slots__:
            if x == "mdp_ids":
                continue  # Torch does not support tensors of strings
            t = getattr(self, x)
            if t is not None:
                assert isinstance(t, torch.Tensor), (
                    x + " is not a torch tensor (is " + str(type(t)) + ")"
                )
                if x == "possible_next_actions_lengths":
                    setattr(self, x, t.type(dtype).long())
                else:
                    setattr(self, x, t.type(dtype))
