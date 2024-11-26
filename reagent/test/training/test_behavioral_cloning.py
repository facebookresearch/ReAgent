#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest
from enum import Enum

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from reagent.core import types as rlt
from reagent.models.dqn import FullyConnectedDQN
from reagent.optimizer.union import classes, Optimizer__Union
from reagent.training.behavioral_cloning_trainer import BehavioralCloningTrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

SEED = 0


class SyntheticType(Enum):
    ACTION_TYPE = "one-hot"  # support 'one-hot'


def get_dummy_batch(action_type, num_batches):
    if action_type == "one-hot":
        action = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        raise TypeError("the actions (labels) should be one-hot")

    possible_actions_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            # 1 means no mask. This mask keeps the label position (diagonal position) and some other position
        ]
    )

    batches = [None for _ in range(num_batches)]
    for i in range(num_batches):
        state = torch.tensor(
            [
                [+0.1, +0.2, +0.3, +0.4, +0.5, +0.6, +0.7, +0.8],
                [+0.1, +0.2, +0.3, +0.4, -0.5, -0.6, -0.7, -0.8],
                [-0.1, -0.2, -0.3, -0.4, +0.5, +0.6, +0.7, +0.8],
                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
            ]
        )
        # 8*1 float embedding
        # -------- means label=0
        # ----++++ means label=1
        # ++++---- means label=2
        # ++++++++ means label=3
        state = state + (1e-8**0.5) * torch.rand_like(state)  # add rand noise
        i_th_training_batch = rlt.BehavioralCloningModelInput(
            state=rlt.FeatureData(float_features=state),
            action=action,
            possible_actions_mask=possible_actions_mask,
        )
        batches[i] = i_th_training_batch
    return batches


def create_synthetic_data(
    num_batches_train: int, num_batches_eval: int
) -> rlt.BehavioralCloningModelInput:
    train_batches = get_dummy_batch(
        action_type=SyntheticType.ACTION_TYPE.value, num_batches=num_batches_train
    )
    train_dataloader = DataLoader(train_batches, collate_fn=lambda x: x[0])

    eval_batches = get_dummy_batch(
        action_type=SyntheticType.ACTION_TYPE.value, num_batches=num_batches_eval
    )
    eval_dataloader = DataLoader(eval_batches, collate_fn=lambda x: x[0])

    # pyre-fixme[7]: Expected `BehavioralCloningModelInput` but got
    #  `Tuple[DataLoader[Any], DataLoader[Any]]`.
    return train_dataloader, eval_dataloader  # list of BehavioralCloningModelInput


def train_bc_model(train_dataloader, num_epochs) -> pl.LightningModule:
    bc_net = FullyConnectedDQN(
        state_dim=8,  # input
        action_dim=4,  # output
        sizes=[7, 6, 5],  # hidden layers
        activations=["relu", "relu", "relu"],
    )

    # pyre-fixme[28]: Unexpected keyword argument `Adam`.
    optimizer = Optimizer__Union(Adam=classes["Adam"]())
    bc_trainer = BehavioralCloningTrainer(bc_net=bc_net, optimizer=optimizer)
    pl_trainer = pl.Trainer(max_epochs=num_epochs, deterministic=True)
    pl_trainer.fit(bc_trainer, train_dataloader)
    return bc_trainer


def validation_prob_vs_label(
    bc_trainer: pl.LightningModule,
    batch: rlt.BehavioralCloningModelInput,
    batch_idx: int,
):
    # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
    masked_logits = bc_trainer.bc_net(
        batch.state,
        batch.possible_actions_mask,
    )
    labels = batch.action
    probs = torch.nn.functional.softmax(masked_logits)
    assert torch.allclose(labels.double(), probs.double(), atol=1e-1)
    return


def eval_bc_model(eval_dataloader, bc_trainer) -> torch.Tensor:
    total_xentropy_loss = 0
    for batch_idx, batch in enumerate(eval_dataloader):
        xentropy_loss = bc_trainer.validation_step(batch, batch_idx)
        total_xentropy_loss += xentropy_loss
    N_eval = len(eval_dataloader)
    eval_xentropy_loss = total_xentropy_loss / N_eval

    # at the last batch, check whether probs matches labels
    # pyre-fixme[61]: `batch` is undefined, or not always defined.
    # pyre-fixme[61]: `batch_idx` is undefined, or not always defined.
    validation_prob_vs_label(bc_trainer, batch, batch_idx)
    # pyre-fixme[7]: Expected `Tensor` but got `float`.
    return eval_xentropy_loss


class TestBehavioralCloning(unittest.TestCase):
    def setUp(self):
        seed_everything(1)

    def test_behavioral_cloning_v0(self):
        NUM_TRAIN_BATCH, NUM_EVAL_BATCH = 200, 200
        train_dataloader, eval_dataloader = create_synthetic_data(
            num_batches_train=NUM_TRAIN_BATCH, num_batches_eval=NUM_EVAL_BATCH
        )
        bc_trainer = train_bc_model(train_dataloader=train_dataloader, num_epochs=4)
        eval_loss = eval_bc_model(
            eval_dataloader=eval_dataloader, bc_trainer=bc_trainer
        )
        logger.info(f"eval_loss={eval_loss}")
        assert abs(eval_loss) < 0.1
