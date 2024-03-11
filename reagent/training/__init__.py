#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.training.behavioral_cloning_trainer import BehavioralCloningTrainer
from reagent.training.c51_trainer import C51Trainer
from reagent.training.cem_trainer import CEMTrainer
from reagent.training.cfeval import BanditRewardNetTrainer
from reagent.training.discrete_crr_trainer import DiscreteCRRTrainer
from reagent.training.dqn_trainer import DQNTrainer
from reagent.training.multi_stage_trainer import MultiStageTrainer
from reagent.training.parametric_dqn_trainer import ParametricDQNTrainer
from reagent.training.ppo_trainer import PPOTrainer
from reagent.training.qrdqn_trainer import QRDQNTrainer
from reagent.training.reagent_lightning_module import (
    ReAgentLightningModule,
    StoppingEpochCallback,
)
from reagent.training.reinforce_trainer import ReinforceTrainer
from reagent.training.reward_network_trainer import RewardNetTrainer
from reagent.training.sac_trainer import SACTrainer
from reagent.training.slate_q_trainer import SlateQTrainer
from reagent.training.td3_trainer import TD3Trainer
from reagent.training.world_model.mdnrnn_trainer import MDNRNNTrainer

from .parameters import (
    BehavioralCloningTrainerParameters,
    C51TrainerParameters,
    CRRTrainerParameters,
    DQNTrainerParameters,
    ParametricDQNTrainerParameters,
    PPOTrainerParameters,
    QRDQNTrainerParameters,
    ReinforceTrainerParameters,
    RewardNetworkTrainerParameters,
    SACTrainerParameters,
    Seq2SlateTrainerParameters,
    SlateQTrainerParameters,
    TD3TrainerParameters,
)


__all__ = [
    "BehavioralCloningTrainer",
    "BanditRewardNetTrainer",
    "C51Trainer",
    "CEMTrainer",
    "DQNTrainer",
    "MultiStageTrainer",
    "MDNRNNTrainer",
    "ParametricDQNTrainer",
    "QRDQNTrainer",
    "SACTrainer",
    "SlateQTrainer",
    "TD3Trainer",
    "DiscreteCRRTrainer",
    "RewardNetTrainer",
    "C51TrainerParameters",
    "DQNTrainerParameters",
    "ParametricDQNTrainerParameters",
    "QRDQNTrainerParameters",
    "SACTrainerParameters",
    "SlateQTrainerParameters",
    "TD3TrainerParameters",
    "CRRTrainerParameters",
    "RewardNetworkTrainerParameters",
    "Seq2SlateTrainerParameters",
    "ReAgentLightningModule",
    "StoppingEpochCallback",
    "ReinforceTrainer",
    "ReinforceTrainerParameters",
    "PPOTrainer",
    "PPOTrainerParameters",
    "BehavioralCloningTrainerParameters",
]

if IS_FB_ENVIRONMENT:
    from reagent.training.fb.signal_loss_reward_decomp_trainer import (  # noqa
        SignalLossRewardDecompTrainer,
    )

    __all__.append("SignalLossRewardDecompTrainer")
