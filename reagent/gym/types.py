#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Please DO NOT import gym in here. We might have installation without gym depending on
# this module for typing

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional

import reagent.types as rlt
import torch


@dataclass
class Transition(rlt.BaseDataClass):
    mdp_id: int
    sequence_number: int
    observation: Any
    action: Any
    reward: float
    terminal: bool
    log_prob: Optional[float] = None
    possible_actions: Optional[List[int]] = None
    possible_actions_mask: Optional[List[int]] = None

    # Same as asdict but filters out none values.
    def asdict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def get_optional_fields(cls) -> List[str]:
    """ return list of optional annotated fields """
    ret: List[str] = []
    for f in fields(cls):
        # Check if exactly two arguments exists and one of them are None type
        if hasattr(f.type, "__args__") and type(None) in f.type.__args__:
            ret.append(f.name)
    return ret


@dataclass
class Trajectory(rlt.BaseDataClass):
    transitions: List[Transition] = field(default_factory=list)

    def __post_init__(self):
        self.optional_field_exist: Dict[str, bool] = {
            f: False for f in get_optional_fields(Transition)
        }

    def __len__(self):
        return len(self.transitions)

    def add_transition(self, transition: Transition):
        if len(self) == 0:
            # remember which optional fields should be filled
            for f in self.optional_field_exist:
                val = getattr(transition, f, None)
                if val is not None:
                    self.optional_field_exist[f] = True

        # check that later additions also fill the same optional fields
        for f, should_exist in self.optional_field_exist.items():
            val = getattr(transition, f, None)
            if (val is not None) != should_exist:
                raise ValueError(
                    f"Field {f} given val {val} whereas should_exist is {should_exist}."
                )

        self.transitions.append(transition)

    def __getattr__(self, attr: str):
        ret = []
        for transition in self.transitions:
            ret.append(getattr(transition, attr))
        return ret

    def calculate_cumulative_reward(self, gamma: float = 1.0):
        """ Return (discounted) sum of rewards. """
        num_transitions = len(self)
        assert num_transitions > 0, "called on empty trajectory"
        rewards = self.reward
        discounts = [gamma ** i for i in range(num_transitions)]
        return sum(reward * discount for reward, discount in zip(rewards, discounts))


class Sampler(ABC):
    """Given scores, select the action."""

    @abstractmethod
    def sample_action(self, scores: Any) -> rlt.ActorOutput:
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, scores: Any, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def update(self) -> None:
        """ Call to update internal parameters (e.g. decay epsilon) """
        pass


# From preprocessed observation, produce scores for sampler to select action
Scorer = Callable[[Any], Any]

# Transform ReplayBuffer's transition batch to trainer.train
TrainerPreprocessor = Callable[[Any], rlt.PreprocessedTrainingBatch]


""" Called after env.step(action)
Args: (state, action, reward, terminal, log_prob)
"""
PostStep = Callable[[Transition], None]


@dataclass
class GaussianSamplerScore(rlt.BaseDataClass):
    loc: torch.Tensor
    scale_log: torch.Tensor
