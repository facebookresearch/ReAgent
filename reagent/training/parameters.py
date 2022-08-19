#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.configuration import make_config_class
from reagent.core.types import BaseDataClass

from .behavioral_cloning_trainer import BehavioralCloningTrainer
from .c51_trainer import C51Trainer
from .cb.deep_represent_linucb_trainer import DeepRepresentLinUCBTrainer
from .cb.linucb_trainer import LinUCBTrainer
from .discrete_crr_trainer import DiscreteCRRTrainer
from .dqn_trainer import DQNTrainer
from .parametric_dqn_trainer import ParametricDQNTrainer
from .ppo_trainer import PPOTrainer
from .qrdqn_trainer import QRDQNTrainer
from .ranking.seq2slate_trainer import Seq2SlateTrainer
from .reinforce_trainer import ReinforceTrainer
from .reward_network_trainer import RewardNetTrainer
from .sac_trainer import SACTrainer
from .slate_q_trainer import SlateQTrainer
from .td3_trainer import TD3Trainer


@make_config_class(
    SACTrainer.__init__,
    blocklist=["use_gpu", "actor_network", "q1_network", "q2_network", "value_network"],
)
class SACTrainerParameters:
    pass


@make_config_class(
    TD3Trainer.__init__,
    blocklist=["use_gpu", "actor_network", "q1_network", "q2_network"],
)
class TD3TrainerParameters:
    pass


@make_config_class(
    DiscreteCRRTrainer.__init__,
    blocklist=[
        "use_gpu",
        "actor_network",
        "actor_network_target",
        "q1_network",
        "q1_network_target",
        "reward_network",
        "q2_network",
        "q2_network_target",
        "q_network_cpe",
        "q_network_cpe_target",
        "metrics_to_score",
        "evaluation",
    ],
)
class CRRTrainerParameters:
    pass


@make_config_class(
    SlateQTrainer.__init__, blocklist=["use_gpu", "q_network", "q_network_target"]
)
class SlateQTrainerParameters:
    pass


@make_config_class(
    ParametricDQNTrainer.__init__,
    blocklist=["use_gpu", "q_network", "q_network_target", "reward_network"],
)
class ParametricDQNTrainerParameters:
    pass


@make_config_class(
    DQNTrainer.__init__,
    blocklist=[
        "use_gpu",
        "q_network",
        "q_network_target",
        "reward_network",
        "q_network_cpe",
        "q_network_cpe_target",
        "metrics_to_score",
        "imitator",
        "loss_reporter",
        "evaluation",
    ],
)
class DQNTrainerParameters:
    pass


@make_config_class(
    QRDQNTrainer.__init__,
    blocklist=[
        "use_gpu",
        "q_network",
        "q_network_target",
        "metrics_to_score",
        "reward_network",
        "q_network_cpe",
        "q_network_cpe_target",
        "loss_reporter",
        "evaluation",
    ],
)
class QRDQNTrainerParameters:
    pass


@make_config_class(
    C51Trainer.__init__,
    blocklist=[
        "use_gpu",
        "q_network",
        "q_network_target",
        "metrics_to_score",
        "loss_reporter",
        "evaluation",
    ],
)
class C51TrainerParameters:
    pass


@make_config_class(RewardNetTrainer.__init__, blocklist=["reward_net"])
class RewardNetworkTrainerParameters:
    pass


@make_config_class(BehavioralCloningTrainer.__init__, blocklist=["bc_net"])
class BehavioralCloningTrainerParameters:
    pass


@make_config_class(
    Seq2SlateTrainer.__init__,
    blocklist=[
        "use_gpu",
        "seq2slate_net",
        "baseline_net",
        "baseline_warmup_num_batches",
    ],
)
class Seq2SlateTrainerParameters(BaseDataClass):
    pass


@make_config_class(
    ReinforceTrainer.__init__,
    blocklist=[
        "policy",
        "value_net",
    ],
)
class ReinforceTrainerParameters:
    pass


@make_config_class(
    PPOTrainer.__init__,
    blocklist=[
        "policy",
        "value_net",
    ],
)
class PPOTrainerParameters:
    pass


@make_config_class(
    LinUCBTrainer.__init__,
    blocklist=[
        "policy",
    ],
)
class LinUCBTrainerParameters:
    pass


@make_config_class(
    DeepRepresentLinUCBTrainer.__init__,
    blocklist=[
        "policy",
    ],
)
class DeepRepresentLinUCBTrainerParameters:
    pass
