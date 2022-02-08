#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import random
import unittest
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from parameterized import parameterized
from reagent.core import types as rlt
from reagent.core.parameters import (
    NormalizationData,
    NormalizationParameters,
    ProblemDomain,
    Seq2RewardTrainerParameters,
)
from reagent.gym.envs import Gym
from reagent.gym.utils import create_df_from_replay_buffer
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.net_builder.value.fully_connected import FullyConnected
from reagent.prediction.predictor_wrapper import (
    Seq2RewardWithPreprocessor,
    Seq2RewardPlanShortSeqWithPreprocessor,
    FAKE_STATE_ID_LIST_FEATURES,
    FAKE_STATE_ID_SCORE_LIST_FEATURES,
)
from reagent.preprocessing.identify_types import DO_NOT_PREPROCESS
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.training.utils import gen_permutations
from reagent.training.world_model.compress_model_trainer import CompressModelTrainer
from reagent.training.world_model.seq2reward_trainer import get_Q, Seq2RewardTrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

SEED = 0
STRING_GAME_TESTS = [(False,), (True,)]


class FakeStepPredictionNetwork(nn.Module):
    def __init__(self, look_ahead_steps):
        super().__init__()
        self.look_ahead_steps = look_ahead_steps

    def forward(self, state: torch.Tensor):
        """
        Given the current state, predict the probability of
        experiencing next n steps (1 <=n <= look_ahead_steps)

        For the test purpose, it outputs fixed fake numbers
        """
        batch_size, _ = state.shape
        return torch.ones(batch_size, self.look_ahead_steps).float()


class FakeSeq2RewardNetwork(nn.Module):
    def forward(
        self,
        state: rlt.FeatureData,
        action: rlt.FeatureData,
        valid_reward_len: Optional[torch.Tensor] = None,
    ):
        """
        Mimic I/O of Seq2RewardNetwork but return fake reward
        Reward is the concatenation of action indices, independent
        of state.

        For example, when seq_len = 3, batch_size = 1, action_num = 2,
        acc_reward = tensor(
            [[  0.],
            [  1.],
            [ 10.],
            [ 11.],
            [100.],
            [101.],
            [110.],
            [111.]]
        )

        Input action shape: seq_len, batch_size, num_action
        Output acc_reward shape: batch_size, 1
        """
        # pyre-fixme[9]: action has type `FeatureData`; used as `Tensor`.
        action = action.float_features.transpose(0, 1)
        action_indices = torch.argmax(action, dim=2).tolist()
        acc_reward = torch.tensor(
            list(map(lambda x: float("".join(map(str, x))), action_indices))
        ).reshape(-1, 1)
        logger.info(f"acc_reward: {acc_reward}")
        return rlt.Seq2RewardOutput(acc_reward=acc_reward)


def create_string_game_data(
    dataset_size=10000, training_data_ratio=0.9, filter_short_sequence=False
):
    SEQ_LEN = 6
    NUM_ACTION = 2
    NUM_MDP_PER_BATCH = 5

    env = Gym(env_name="StringGame-v0", set_max_steps=SEQ_LEN)
    df = create_df_from_replay_buffer(
        env=env,
        problem_domain=ProblemDomain.DISCRETE_ACTION,
        desired_size=dataset_size,
        multi_steps=None,
        ds="2020-10-10",
    )

    if filter_short_sequence:
        batch_size = NUM_MDP_PER_BATCH
        time_diff = torch.ones(SEQ_LEN, batch_size)
        valid_step = SEQ_LEN * torch.ones(batch_size, dtype=torch.int64)[:, None]
        not_terminal = torch.Tensor(
            [0 if i == SEQ_LEN - 1 else 1 for i in range(SEQ_LEN)]
        )
        not_terminal = torch.transpose(not_terminal.tile(NUM_MDP_PER_BATCH, 1), 0, 1)
    else:
        batch_size = NUM_MDP_PER_BATCH * SEQ_LEN
        time_diff = torch.ones(SEQ_LEN, batch_size)
        valid_step = torch.arange(SEQ_LEN, 0, -1).tile(NUM_MDP_PER_BATCH)[:, None]
        not_terminal = torch.transpose(
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=-1).tile(
                NUM_MDP_PER_BATCH, 1
            ),
            0,
            1,
        )

    num_batches = int(dataset_size / SEQ_LEN / NUM_MDP_PER_BATCH)
    batches = [None for _ in range(num_batches)]
    batch_count, batch_seq_count = 0, 0
    batch_reward = torch.zeros(SEQ_LEN, batch_size)
    batch_action = torch.zeros(SEQ_LEN, batch_size, NUM_ACTION)
    batch_state = torch.zeros(SEQ_LEN, batch_size, NUM_ACTION)
    for mdp_id in sorted(set(df.mdp_id)):
        mdp = df[df["mdp_id"] == mdp_id].sort_values("sequence_number", ascending=True)
        if len(mdp) != SEQ_LEN:
            continue

        all_step_reward = torch.Tensor(list(mdp["reward"]))
        all_step_state = torch.Tensor([list(s.values()) for s in mdp["state_features"]])
        all_step_action = torch.zeros_like(all_step_state)
        all_step_action[torch.arange(SEQ_LEN), [int(a) for a in mdp["action"]]] = 1.0

        for j in range(SEQ_LEN):
            if filter_short_sequence and j > 0:
                break

            reward = torch.zeros_like(all_step_reward)
            reward[: SEQ_LEN - j] = all_step_reward[-(SEQ_LEN - j) :]
            batch_reward[:, batch_seq_count] = reward

            state = torch.zeros_like(all_step_state)
            state[: SEQ_LEN - j] = all_step_state[-(SEQ_LEN - j) :]
            batch_state[:, batch_seq_count] = state

            action = torch.zeros_like(all_step_action)
            action[: SEQ_LEN - j] = all_step_action[-(SEQ_LEN - j) :]
            batch_action[:, batch_seq_count] = action

            batch_seq_count += 1

        if batch_seq_count == batch_size:
            batches[batch_count] = rlt.MemoryNetworkInput(
                reward=batch_reward,
                action=rlt.FeatureData(float_features=batch_action),
                state=rlt.FeatureData(float_features=batch_state),
                next_state=rlt.FeatureData(
                    float_features=torch.zeros_like(batch_state)
                ),  # fake, not used anyway
                not_terminal=not_terminal,
                time_diff=time_diff,
                valid_step=valid_step,
                step=None,
            )
            batch_count += 1
            batch_seq_count = 0
            batch_reward = torch.zeros_like(batch_reward)
            batch_action = torch.zeros_like(batch_action)
            batch_state = torch.zeros_like(batch_state)
    assert batch_count == num_batches

    num_training_batches = int(training_data_ratio * num_batches)
    training_data = DataLoader(
        batches[:num_training_batches], collate_fn=lambda x: x[0]
    )
    eval_data = DataLoader(batches[num_training_batches:], collate_fn=lambda x: x[0])
    return training_data, eval_data


def train_seq2reward_model(training_data, learning_rate=0.01, num_epochs=5):
    SEQ_LEN, batch_size, NUM_ACTION = next(
        iter(training_data)
    ).action.float_features.shape
    assert SEQ_LEN == 6 and NUM_ACTION == 2

    seq2reward_network = Seq2RewardNetwork(
        state_dim=NUM_ACTION,
        action_dim=NUM_ACTION,
        num_hiddens=64,
        num_hidden_layers=2,
    )

    trainer_param = Seq2RewardTrainerParameters(
        learning_rate=learning_rate,
        multi_steps=SEQ_LEN,
        action_names=["0", "1"],
        gamma=1.0,
        view_q_value=True,
    )

    trainer = Seq2RewardTrainer(
        seq2reward_network=seq2reward_network, params=trainer_param
    )

    pl.seed_everything(SEED)
    pl_trainer = pl.Trainer(max_epochs=num_epochs, deterministic=True)
    pl_trainer.fit(trainer, training_data)

    return trainer


def eval_seq2reward_model(eval_data, seq2reward_trainer):
    SEQ_LEN, batch_size, NUM_ACTION = next(iter(eval_data)).action.float_features.shape

    initial_state = torch.Tensor([[0, 0]])
    initial_state_q_values = torch.squeeze(
        get_Q(
            seq2reward_trainer.seq2reward_network,
            initial_state,
            seq2reward_trainer.all_permut,
        )
    )

    total_mse_loss = 0
    total_q_values = torch.zeros(NUM_ACTION)
    total_action_distribution = torch.zeros(NUM_ACTION)
    for idx, batch in enumerate(eval_data):
        (
            mse_loss,
            _,
            q_values,
            action_distribution,
        ) = seq2reward_trainer.validation_step(batch, idx)
        total_mse_loss += mse_loss
        total_q_values += torch.tensor(q_values)
        total_action_distribution += torch.tensor(action_distribution)

    N_eval = len(eval_data)
    eval_mse_loss = total_mse_loss / N_eval
    eval_q_values = total_q_values / N_eval
    eval_action_distribution = total_action_distribution / N_eval

    return (
        initial_state_q_values,
        eval_mse_loss,
        eval_q_values,
        eval_action_distribution,
    )


def train_seq2reward_compress_model(
    training_data, seq2reward_network, learning_rate=0.1, num_epochs=5
):
    SEQ_LEN, batch_size, NUM_ACTION = next(
        iter(training_data)
    ).action.float_features.shape
    assert SEQ_LEN == 6 and NUM_ACTION == 2

    compress_net_builder = FullyConnected(sizes=[8, 8])
    state_normalization_data = NormalizationData(
        dense_normalization_parameters={
            0: NormalizationParameters(feature_type=DO_NOT_PREPROCESS),
            1: NormalizationParameters(feature_type=DO_NOT_PREPROCESS),
        }
    )
    compress_model_network = compress_net_builder.build_value_network(
        state_normalization_data,
        output_dim=NUM_ACTION,
    )

    trainer_param = Seq2RewardTrainerParameters(
        learning_rate=0.0,
        multi_steps=SEQ_LEN,
        action_names=["0", "1"],
        compress_model_learning_rate=learning_rate,
        gamma=1.0,
        view_q_value=True,
    )

    trainer = CompressModelTrainer(
        compress_model_network=compress_model_network,
        seq2reward_network=seq2reward_network,
        params=trainer_param,
    )

    pl.seed_everything(SEED)
    pl_trainer = pl.Trainer(max_epochs=num_epochs, deterministic=True)
    pl_trainer.fit(trainer, training_data)

    return trainer


def eval_seq2reward_compress_model(eval_data, compress_model_trainer):
    SEQ_LEN, batch_size, NUM_ACTION = next(iter(eval_data)).action.float_features.shape
    total_mse_loss = 0
    total_q_values = torch.zeros(NUM_ACTION)
    total_action_distribution = torch.zeros(NUM_ACTION)
    for idx, batch in enumerate(eval_data):
        (
            mse_loss,
            q_values,
            action_distribution,
            _,
        ) = compress_model_trainer.validation_step(batch, idx)
        total_mse_loss += mse_loss
        total_q_values += torch.tensor(q_values)
        total_action_distribution += torch.tensor(action_distribution)

    N_eval = len(eval_data)
    eval_mse_loss = total_mse_loss / N_eval
    eval_q_values = total_q_values / N_eval
    eval_action_distribution = total_action_distribution / N_eval

    return eval_mse_loss, eval_q_values, eval_action_distribution


class TestSeq2Reward(unittest.TestCase):
    def test_seq2reward_with_preprocessor_plan_short_sequence(self):
        self._test_seq2reward_with_preprocessor(plan_short_sequence=True)

    def test_seq2reward_with_preprocessor_plan_full_sequence(self):
        self._test_seq2reward_with_preprocessor(plan_short_sequence=False)

    def _test_seq2reward_with_preprocessor(self, plan_short_sequence):
        state_dim = 4
        action_dim = 2
        seq_len = 3
        model = FakeSeq2RewardNetwork()
        state_normalization_parameters = {
            i: NormalizationParameters(
                feature_type=DO_NOT_PREPROCESS, mean=0.0, stddev=1.0
            )
            for i in range(1, state_dim)
        }
        state_preprocessor = Preprocessor(state_normalization_parameters, False)

        if plan_short_sequence:
            step_prediction_model = FakeStepPredictionNetwork(seq_len)
            model_with_preprocessor = Seq2RewardPlanShortSeqWithPreprocessor(
                model,
                step_prediction_model,
                state_preprocessor,
                seq_len,
                action_dim,
            )
        else:
            model_with_preprocessor = Seq2RewardWithPreprocessor(
                model,
                state_preprocessor,
                seq_len,
                action_dim,
            )
        input_prototype = rlt.ServingFeatureData(
            float_features_with_presence=state_preprocessor.input_prototype(),
            id_list_features=FAKE_STATE_ID_LIST_FEATURES,
            id_score_list_features=FAKE_STATE_ID_SCORE_LIST_FEATURES,
        )
        q_values = model_with_preprocessor(input_prototype)
        if plan_short_sequence:
            # When planning for 1, 2, and 3 steps ahead,
            # the expected q values are respectively:
            # [0, 1], [1, 11], [11, 111]
            # Weighting the expected q values by predicted step
            # probabilities [0.33, 0.33, 0.33], we have [4, 41]
            expected_q_values = torch.tensor([[4.0, 41.0]])
        else:
            expected_q_values = torch.tensor([[11.0, 111.0]])
        assert torch.all(expected_q_values == q_values)

    def test_get_Q(self):
        NUM_ACTION = 2
        MULTI_STEPS = 3
        BATCH_SIZE = 2
        STATE_DIM = 4
        all_permut = gen_permutations(MULTI_STEPS, NUM_ACTION)
        seq2reward_network = FakeSeq2RewardNetwork()
        state = torch.zeros(BATCH_SIZE, STATE_DIM)
        q_values = get_Q(seq2reward_network, state, all_permut)
        expected_q_values = torch.tensor([[11.0, 111.0], [11.0, 111.0]])
        logger.info(f"q_values: {q_values}")
        assert torch.all(expected_q_values == q_values)

    def test_gen_permutations_seq_len_1_action_6(self):
        SEQ_LEN = 1
        NUM_ACTION = 6
        expected_outcome = torch.tensor([[0], [1], [2], [3], [4], [5]])
        self._test_gen_permutations(SEQ_LEN, NUM_ACTION, expected_outcome)

    def test_gen_permutations_seq_len_3_num_action_2(self):
        SEQ_LEN = 3
        NUM_ACTION = 2
        expected_outcome = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        self._test_gen_permutations(SEQ_LEN, NUM_ACTION, expected_outcome)

    def _test_gen_permutations(self, SEQ_LEN, NUM_ACTION, expected_outcome):
        # expected shape: SEQ_LEN, PERM_NUM, ACTION_DIM
        result = gen_permutations(SEQ_LEN, NUM_ACTION)
        assert result.shape == (SEQ_LEN, NUM_ACTION ** SEQ_LEN, NUM_ACTION)
        outcome = torch.argmax(result.transpose(0, 1), dim=-1)
        assert torch.all(outcome == expected_outcome)

    @parameterized.expand(STRING_GAME_TESTS)
    @unittest.skipIf("SANDCASTLE" in os.environ, "Skipping long test on sandcastle.")
    def test_seq2reward_on_string_game_v0(self, filter_short_sequence):
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        training_data, eval_data = create_string_game_data(
            filter_short_sequence=filter_short_sequence
        )
        seq2reward_trainer = train_seq2reward_model(training_data)
        (
            initial_state_q_values,
            eval_mse_loss,
            eval_q_values,
            eval_action_distribution,
        ) = eval_seq2reward_model(eval_data, seq2reward_trainer)

        assert abs(initial_state_q_values[0].item() - 10) < 1.0
        assert abs(initial_state_q_values[1].item() - 5) < 1.0

        if filter_short_sequence:
            assert eval_mse_loss < 0.1
        else:
            # Same short sequences may have different total rewards due to the missing
            # states and actions in previous steps, so the trained network is not able
            # to reduce the mse loss to values close to zero.
            assert eval_mse_loss < 10

        compress_model_trainer = train_seq2reward_compress_model(
            training_data, seq2reward_trainer.seq2reward_network
        )
        (
            compress_eval_mse_loss,
            compress_eval_q_values,
            compress_eval_action_distribution,
        ) = eval_seq2reward_compress_model(eval_data, compress_model_trainer)

        assert compress_eval_mse_loss < 1e-5
        assert torch.all(eval_q_values - compress_eval_q_values < 1e-5)
        assert torch.all(
            eval_action_distribution - compress_eval_action_distribution < 1e-5
        )
