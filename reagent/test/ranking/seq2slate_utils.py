#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
import tempfile
from itertools import permutations

import pytorch_lightning as pl
import reagent.core.types as rlt
import torch
import torch.nn as nn
from reagent.core.parameters import Seq2SlateParameters
from reagent.core.parameters_seq2slate import LearningMethod, SimulationParameters
from reagent.core.torch_utils import gather
from reagent.model_utils.seq2slate_utils import Seq2SlateOutputArch
from reagent.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.training.ranking.seq2slate_sim_trainer import Seq2SlateSimulationTrainer
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


MODEL_TRANSFORMER = "transformer"
ON_POLICY = "on_policy"
OFF_POLICY = "off_policy"
SIMULATION = "simulation"


class TSPRewardModel(nn.Module):
    def forward(self, state, candidates, ranked_cities, src_src_mask, tgt_out_idx):
        reward = compute_reward(ranked_cities)
        # negate because we want to minimize
        return -reward


def post_preprocess_batch(seq2slate_net, candidate_num, batch, device, epoch):
    model_propensity, model_action, reward = rank_on_policy_and_eval(
        seq2slate_net, batch, candidate_num, greedy=False
    )
    batch = rlt.PreprocessedRankingInput.from_input(
        state=batch.state.float_features,
        candidates=batch.src_seq.float_features,
        device=device,
        action=model_action,
        logged_propensities=model_propensity,
        # negate because we want to minimize
        slate_reward=-reward,
    )
    logger.info(f"Epoch {epoch} mean on_policy reward: {torch.mean(reward)}")
    logger.info(f"Epoch {epoch} mean model_propensity: {torch.mean(model_propensity)}")
    return batch


class Seq2SlateOnPolicyTrainer(Seq2SlateTrainer):
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        new_batch = post_preprocess_batch(
            self.seq2slate_net,
            self.seq2slate_net.max_src_seq_len,
            batch,
            batch.state.float_features.device,
            self.current_epoch,
        )
        for attr in dir(new_batch):
            if not callable(getattr(new_batch, attr)) and not attr.startswith("__"):
                setattr(batch, attr, getattr(new_batch, attr))
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)


def create_trainer(
    seq2slate_net,
    learning_method,
    batch_size,
    learning_rate,
    policy_gradient_interval,
    device,
):
    if learning_method == ON_POLICY:
        seq2slate_params = Seq2SlateParameters(
            on_policy=True, learning_method=LearningMethod.REINFORCEMENT_LEARNING
        )
        trainer_cls = Seq2SlateOnPolicyTrainer
    elif learning_method == OFF_POLICY:
        seq2slate_params = Seq2SlateParameters(
            on_policy=False,
            learning_method=LearningMethod.REINFORCEMENT_LEARNING,
        )
        trainer_cls = Seq2SlateTrainer
    elif learning_method == SIMULATION:
        temp_reward_model_path = tempfile.mkstemp(suffix=".pt")[1]
        reward_model = torch.jit.script(TSPRewardModel())
        torch.jit.save(reward_model, temp_reward_model_path)
        seq2slate_params = Seq2SlateParameters(
            on_policy=True,
            learning_method=LearningMethod.SIMULATION,
            simulation=SimulationParameters(
                reward_name_weight={"tour_length": 1.0},
                reward_name_power={"tour_length": 1.0},
                reward_name_path={"tour_length": temp_reward_model_path},
            ),
        )
        trainer_cls = Seq2SlateSimulationTrainer

    param_dict = {
        "seq2slate_net": seq2slate_net,
        "params": seq2slate_params,
        "policy_optimizer": Optimizer__Union.default(lr=learning_rate),
        "print_interval": 1,
        "policy_gradient_interval": policy_gradient_interval,
    }
    return trainer_cls(**param_dict)


def create_seq2slate_net(
    model_str,
    candidate_num,
    candidate_dim,
    hidden_size,
    output_arch,
    temperature,
    device,
):
    if model_str == MODEL_TRANSFORMER:
        return Seq2SlateTransformerNet(
            state_dim=1,
            candidate_dim=candidate_dim,
            num_stacked_layers=2,
            num_heads=2,
            dim_model=hidden_size,
            dim_feedforward=hidden_size,
            max_src_seq_len=candidate_num,
            max_tgt_seq_len=candidate_num,
            output_arch=output_arch,
            temperature=temperature,
            state_embed_dim=1,
        ).to(device)
    else:
        raise NotImplementedError(f"unknown model type {model_str}")


FIX_CANDIDATES = None


@torch.no_grad()
def create_batch(
    batch_size,
    candidate_num,
    candidate_dim,
    device,
    learning_method,
    diverse_input=False,
):
    # fake state, we only use candidates
    state = torch.zeros(batch_size, 1)
    if diverse_input:
        # city coordinates are spread in [0, 4]
        candidates = torch.randint(
            5, (batch_size, candidate_num, candidate_dim)
        ).float()
    else:
        # every training data has the same nodes as the input cities
        global FIX_CANDIDATES
        if FIX_CANDIDATES is None or FIX_CANDIDATES.shape != (
            batch_size,
            candidate_num,
            candidate_dim,
        ):
            candidates = torch.randint(
                5, (batch_size, candidate_num, candidate_dim)
            ).float()
            candidates[1:] = candidates[0]
            FIX_CANDIDATES = candidates
        else:
            candidates = FIX_CANDIDATES

    batch_dict = {
        "state": state,
        "candidates": candidates,
        "device": device,
    }
    if learning_method == OFF_POLICY:
        # using data from a uniform sampling policy
        action = torch.stack([torch.randperm(candidate_num) for _ in range(batch_size)])
        propensity = torch.full((batch_size, 1), 1.0 / math.factorial(candidate_num))
        ranked_cities = gather(candidates, action)
        reward = compute_reward(ranked_cities)
        batch_dict["action"] = action
        batch_dict["logged_propensities"] = propensity
        batch_dict["slate_reward"] = -reward

    batch = rlt.PreprocessedRankingInput.from_input(**batch_dict)
    logger.info("Generate one batch")
    return batch


def create_train_and_test_batches(
    batch_size,
    candidate_num,
    candidate_dim,
    device,
    num_train_batches,
    learning_method,
    diverse_input,
):
    train_batches = [
        create_batch(
            batch_size,
            candidate_num,
            candidate_dim,
            device,
            learning_method,
            diverse_input=diverse_input,
        )
        for _ in range(num_train_batches)
    ]

    if diverse_input:
        test_batch = create_batch(
            batch_size,
            candidate_num,
            candidate_dim,
            device,
            learning_method,
            diverse_input=diverse_input,
        )
    else:
        test_batch = train_batches[0]

    return train_batches, test_batch


def compute_reward(ranked_cities):
    assert len(ranked_cities.shape) == 3
    ranked_cities_offset = torch.roll(ranked_cities, shifts=1, dims=1)
    return (
        torch.sqrt(((ranked_cities_offset - ranked_cities) ** 2).sum(-1))
        .sum(-1)
        .unsqueeze(1)
    )


def compute_best_reward(input_cities):
    batch_size, candidate_num, _ = input_cities.shape
    all_perm = torch.tensor(
        list(permutations(torch.arange(candidate_num), candidate_num))
    )
    res = [
        compute_reward(gather(input_cities, perm.repeat(batch_size, 1)))
        for perm in all_perm
    ]
    # res shape: batch_size, num_perm
    res = torch.cat(res, dim=1)
    best_possible_reward = torch.min(res, dim=1).values
    best_possible_reward_mean = torch.mean(best_possible_reward)
    return best_possible_reward_mean


@torch.no_grad()
def rank_on_policy(
    model, batch: rlt.PreprocessedRankingInput, tgt_seq_len: int, greedy: bool
):
    model.eval()
    rank_output = model(
        batch, mode=Seq2SlateMode.RANK_MODE, tgt_seq_len=tgt_seq_len, greedy=greedy
    )
    ranked_slate_prob = rank_output.ranked_per_seq_probs
    ranked_order = rank_output.ranked_tgt_out_idx - 2
    model.train()
    return ranked_slate_prob, ranked_order


@torch.no_grad()
def rank_on_policy_and_eval(
    seq2slate_net, batch: rlt.PreprocessedRankingInput, tgt_seq_len: int, greedy: bool
):
    model_propensity, model_action = rank_on_policy(
        seq2slate_net, batch, tgt_seq_len, greedy=greedy
    )
    ranked_cities = gather(batch.src_seq.float_features, model_action)
    reward = compute_reward(ranked_cities)
    return model_propensity, model_action, reward


def run_seq2slate_tsp(
    model_str,
    batch_size,
    epochs,
    candidate_num,
    num_batches,
    hidden_size,
    diverse_input,
    learning_rate,
    expect_reward_threshold,
    learning_method,
    policy_gradient_interval,
    device,
):
    pl.seed_everything(0)

    candidate_dim = 2
    eval_sample_size = 1

    train_batches, test_batch = create_train_and_test_batches(
        batch_size,
        candidate_num,
        candidate_dim,
        device,
        num_batches,
        learning_method,
        diverse_input,
    )
    best_test_possible_reward = compute_best_reward(test_batch.src_seq.float_features)

    seq2slate_net = create_seq2slate_net(
        model_str,
        candidate_num,
        candidate_dim,
        hidden_size,
        Seq2SlateOutputArch.AUTOREGRESSIVE,
        1.0,
        device,
    )

    trainer = create_trainer(
        seq2slate_net,
        learning_method,
        batch_size,
        learning_rate,
        policy_gradient_interval,
        device,
    )

    def evaluate():
        best_test_reward = torch.full((batch_size,), 1e9).to(device)
        for _ in range(eval_sample_size):
            model_propensities, _, reward = rank_on_policy_and_eval(
                seq2slate_net.to(device), test_batch, candidate_num, greedy=True
            )
            best_test_reward = torch.where(
                reward < best_test_reward, reward, best_test_reward
            )
        logger.info(
            f"Test mean model_propensities {torch.mean(model_propensities)}, "
            f"Test mean reward: {torch.mean(best_test_reward)}, "
            f"best possible reward {best_test_possible_reward}"
        )
        if torch.any(torch.isnan(model_propensities)):
            raise Exception("Model propensities contain NaNs")
        ratio = torch.mean(best_test_reward) / best_test_possible_reward
        return ratio < expect_reward_threshold, ratio

    evaluate()

    training_data = DataLoader(train_batches, collate_fn=lambda x: x[0])
    pl_trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=None if device == torch.device("cpu") else 1,
        logger=False,
    )
    pl_trainer.fit(trainer, training_data)

    result, ratio = evaluate()

    assert result, (
        f"Test failed because it did not reach expected test reward, "
        f"{ratio} > {expect_reward_threshold}."
    )
