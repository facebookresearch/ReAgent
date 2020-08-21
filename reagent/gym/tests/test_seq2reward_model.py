#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import unittest
from typing import Optional

import torch
from reagent.core.types import RewardOptions
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.envs.gym import Gym
from reagent.gym.preprocessors import make_replay_buffer_trainer_preprocessor
from reagent.gym.utils import build_normalizer, fill_replay_buffer
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.training.world_model.seq2reward_trainer import Seq2RewardTrainer
from reagent.workflow.model_managers.union import ModelManager__Union


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

curr_dir = os.path.dirname(__file__)

SEED = 0


def print_seq2reward_losses(epoch, batch_num, losses):
    logger.info(
        f"Printing loss for Epoch {epoch}, Batch {batch_num};\n" f"loss={losses} \n"
    )


def train_seq2reward(
    env: EnvWrapper,
    trainer: Seq2RewardTrainer,
    trainer_preprocessor,
    num_train_transitions: int,
    seq_len: int,
    batch_size: int,
    num_train_epochs: int,
    # for optional validation
    test_replay_buffer=None,
):
    train_replay_buffer = ReplayBuffer(
        replay_capacity=num_train_transitions,
        batch_size=batch_size,
        stack_size=seq_len,
        return_everything_as_stack=True,
    )
    fill_replay_buffer(env, train_replay_buffer, num_train_transitions)
    num_batch_per_epoch = train_replay_buffer.size // batch_size
    logger.info("Made RBs, starting to train now!")
    state_dim = env.observation_space.shape[0]
    for epoch in range(num_train_epochs):
        for i in range(num_batch_per_epoch):
            batch = train_replay_buffer.sample_transition_batch(batch_size=batch_size)
            preprocessed_batch = trainer_preprocessor(batch)
            adhoc_action_padding(preprocessed_batch, state_dim=state_dim)
            losses = trainer.train(preprocessed_batch)
            print_seq2reward_losses(epoch, i, losses)

        # validation
        if test_replay_buffer is not None:
            with torch.no_grad():
                trainer.seq2reward_network.eval()
                test_batch = test_replay_buffer.sample_transition_batch(
                    batch_size=batch_size
                )
                preprocessed_test_batch = trainer_preprocessor(test_batch)
                adhoc_action_padding(preprocessed_test_batch, state_dim=state_dim)
                valid_losses = trainer.get_loss(preprocessed_test_batch)
                print_seq2reward_losses(epoch, "validation", valid_losses)
                trainer.seq2reward_network.train()
    return trainer


def adhoc_action_padding(preprocessed_batch, state_dim):
    # Ad-hoc padding:
    # padding action to zero so that it aligns with the state padding
    # this should be helpful to reduce the confusion during training.
    assert len(preprocessed_batch.state.float_features.size()) == 3
    mask = (
        preprocessed_batch.state.float_features.bool()
        .any(2)
        .int()
        .unsqueeze(2)
        .repeat(1, 1, state_dim)
    )
    assert mask.size() == preprocessed_batch.action.size()
    preprocessed_batch.action = preprocessed_batch.action * mask


def train_seq2reward_and_compute_reward_mse(
    env_name: str,
    model: ModelManager__Union,
    num_train_transitions: int,
    num_test_transitions: int,
    seq_len: int,
    batch_size: int,
    num_train_epochs: int,
    use_gpu: bool,
    saved_seq2reward_path: Optional[str] = None,
):
    """ Train Seq2Reward Network and compute reward mse. """
    env = Gym(env_name=env_name)
    env.seed(SEED)

    manager = model.value
    trainer = manager.initialize_trainer(
        use_gpu=use_gpu,
        reward_options=RewardOptions(),
        normalization_data_map=build_normalizer(env),
    )

    device = "cuda" if use_gpu else "cpu"
    # pyre-fixme[6]: Expected `device` for 2nd param but got `str`.
    trainer_preprocessor = make_replay_buffer_trainer_preprocessor(trainer, device, env)
    test_replay_buffer = ReplayBuffer(
        replay_capacity=num_test_transitions,
        batch_size=batch_size,
        stack_size=seq_len,
        return_everything_as_stack=True,
    )
    fill_replay_buffer(env, test_replay_buffer, num_test_transitions)

    if saved_seq2reward_path is None:
        # train from scratch
        trainer = train_seq2reward(
            env=env,
            trainer=trainer,
            trainer_preprocessor=trainer_preprocessor,
            num_train_transitions=num_train_transitions,
            seq_len=seq_len,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            test_replay_buffer=test_replay_buffer,
        )
    else:
        # load a pretrained model, and just evaluate it
        trainer.seq2reward_network.load_state_dict(torch.load(saved_seq2reward_path))
    state_dim = env.observation_space.shape[0]
    with torch.no_grad():
        trainer.seq2reward_network.eval()
        test_batch = test_replay_buffer.sample_transition_batch(
            batch_size=test_replay_buffer.size
        )
        preprocessed_test_batch = trainer_preprocessor(test_batch)
        adhoc_action_padding(preprocessed_test_batch, state_dim=state_dim)
        losses = trainer.get_loss(preprocessed_test_batch)
        detached_losses = losses.cpu().detach().item()
        trainer.seq2reward_network.train()
    return detached_losses


class TestSeq2Reward(HorizonTestBase):
    @staticmethod
    def verify_result(result: torch.Tensor, mse_threshold: float):
        assert result < mse_threshold, f"mse: {result}, mse_threshold: {mse_threshold}"

    def test_seq2reward(self):
        config_path = "configs/world_model/seq2reward_test.yaml"
        losses = self.run_from_config(
            run_test=train_seq2reward_and_compute_reward_mse,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        TestSeq2Reward.verify_result(losses, 0.001)
        logger.info("Seq2Reward MSE test passes!")


if __name__ == "__main__":
    unittest.main()
