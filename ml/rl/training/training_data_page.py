#!/usr/bin/env python3


class TrainingDataPage(object):
    __slots__ = [
        "states",
        "actions",
        "propensities",
        "rewards",
        "next_states",
        "next_actions",
        "possible_next_actions",
        "possible_next_actions_lengths",
        "reward_timelines",
        "not_terminals",
        "time_diffs",
    ]

    def __init__(
        self,
        states=None,
        actions=None,
        propensities=None,
        rewards=None,
        next_states=None,
        next_actions=None,
        possible_next_actions=None,
        reward_timelines=None,
        not_terminals=None,
        time_diffs=None,
        possible_next_actions_lengths=None,
    ) -> None:
        """
        Creates a TrainingDataPage object.

        In the case where `not_terminals` can be determined by next_actions or
        possible_next_actions, feel free to omit it.
        """
        self.states = states
        self.actions = actions
        self.propensities = propensities
        self.rewards = rewards
        self.next_states = next_states
        self.next_actions = next_actions
        self.possible_next_actions = possible_next_actions
        self.reward_timelines = reward_timelines
        self.not_terminals = not_terminals
        self.time_diffs = time_diffs
        self.possible_next_actions_lengths = possible_next_actions_lengths

    def size(self) -> int:
        if self.states:
            return len(self.states)
        raise Exception("Cannot get size of TrainingDataPage missing states.")

    def get_sub_page(self, start, end):
        if isinstance(self.possible_next_actions, (list, tuple)):
            assert len(self.possible_next_actions) == 2, "Invalid size for pna"
            sub_pna = (
                self.possible_next_actions[0][start:end],
                self.possible_next_actions[1][start:end],
            )
        else:
            sub_pna = (self.possible_next_actions[start:end],)

        return TrainingDataPage(
            self.states[start:end],
            self.actions[start:end],
            self.propensities[start:end],
            self.rewards[start:end],
            self.next_states[start:end],
            None if self.next_actions is None else self.next_actions[start:end],
            None if self.possible_next_actions is None else sub_pna,
            None if self.reward_timelines is None else self.reward_timelines[start:end],
            None if self.not_terminals is None else self.not_terminals[start:end],
            None if self.time_diffs is None else self.time_diffs[start:end],
            None
            if self.possible_next_actions_lengths is None
            else self.possible_next_actions_lengths[start:end],
        )
