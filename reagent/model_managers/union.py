#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Register all ModelManagers. Must import them before filling union. """

from typing import Optional

from reagent.core.tagged_union import TaggedUnion

from .actor_critic import SAC as SACClass, TD3 as TD3Class
from .discrete import (
    DiscreteC51DQN as DiscreteC51DQNClass,
    DiscreteCRR as DiscreteCRRClass,
    DiscreteDQN as DiscreteDQNClass,
    DiscreteQRDQN as DiscreteQRDQNClass,
)
from .model_based import (
    CrossEntropyMethod as CrossEntropyMethodClass,
    Seq2RewardModel as Seq2RewardModelClass,
    WorldModel as WorldModelClass,
)
from .parametric import ParametricDQN as ParametricDQNClass
from .policy_gradient import PPO as PPOClass, Reinforce as ReinforceClass
from .ranking import SlateQ as SlateQClass


class ModelManager__Union(TaggedUnion):
    SAC: Optional[SACClass] = None
    TD3: Optional[TD3Class] = None

    DiscreteC51DQN: Optional[DiscreteC51DQNClass] = None
    DiscreteCRR: Optional[DiscreteCRRClass] = None
    DiscreteDQN: Optional[DiscreteDQNClass] = None
    DiscreteQRDQN: Optional[DiscreteQRDQNClass] = None

    CrossEntropyMethod: Optional[CrossEntropyMethodClass] = None
    Seq2RewardModel: Optional[Seq2RewardModelClass] = None
    WorldModel: Optional[WorldModelClass] = None

    ParametricDQN: Optional[ParametricDQNClass] = None

    PPO: Optional[PPOClass] = None
    Reinforce: Optional[ReinforceClass] = None

    SlateQ: Optional[SlateQClass] = None

    __annotations__ = {
        "SAC": Optional[SACClass],
        "TD3": Optional[TD3Class],
        "DiscreteC51DQN": Optional[DiscreteC51DQNClass],
        "DiscreteCRR": Optional[DiscreteCRRClass],
        "DiscreteDQN": Optional[DiscreteDQNClass],
        "DiscreteQRDQN": Optional[DiscreteQRDQNClass],
        "CrossEntropyMethod": Optional[CrossEntropyMethodClass],
        "Seq2RewardModel": Optional[Seq2RewardModelClass],
        "WorldModel": Optional[WorldModelClass],
        "ParametricDQN": Optional[ParametricDQNClass],
        "PPO": Optional[PPOClass],
        "Reinforce": Optional[ReinforceClass],
        "SlateQ": Optional[SlateQClass],
    }
