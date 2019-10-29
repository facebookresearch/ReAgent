#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import time

import ml.rl.types as rlt
import torch
from ml.rl.models.seq2slate import LOG_PROB_MODE, BaselineNet, Seq2SlateTransformerNet
from ml.rl.parameters import EvaluationParameters, Seq2SlateTransformerParameters
from ml.rl.training.ranking.ranking_trainer import RankingTrainer


logger = logging.getLogger(__name__)


class Seq2SlateTrainer(RankingTrainer):
    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        baseline_net: BaselineNet,
        parameters: Seq2SlateTransformerParameters,
        eval_parameters: EvaluationParameters,
        minibatch_size: int,
        use_gpu: bool = False,
    ) -> None:
        self.parameters = parameters
        self.eval_parameters = eval_parameters
        self.use_gpu = use_gpu
        self.seq2slate_net = seq2slate_net
        self.baseline_net = baseline_net
        self.minibatch_size = minibatch_size
        self.minibatch = 0
        self.rl_opt = torch.optim.Adam(
            self.seq2slate_net.parameters(), lr=1e-3, amsgrad=True
        )
        self.baseline_opt = torch.optim.Adam(
            self.baseline_net.parameters(), lr=1e-3, amsgrad=True
        )

    def warm_start_components(self):
        components = ["seq2slate_net", "baseline_net"]
        return components

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        t1 = time.time()
        assert (
            type(training_batch) is rlt.PreprocessedTrainingBatch
            and type(training_batch.training_input) is rlt.PreprocessedRankingInput
        )
        training_input = training_batch.training_input
        reward = training_input.slate_reward
        batch_size = training_input.state.float_features.shape[0]

        # Train baseline
        b = self.baseline_net(training_input).squeeze()
        baseline_loss = 1.0 / batch_size * torch.sum((b - reward) ** 2)
        baseline_loss.backward()
        self.baseline_opt.step()
        self.baseline_opt.zero_grad()

        # Train Seq2Slate using REINFORCE
        # log probs of tgt seqs
        log_probs = self.seq2slate_net(training_input, mode=LOG_PROB_MODE).log_probs
        b = b.detach()
        assert b.shape == reward.shape == log_probs.shape
        assert not b.requires_grad and log_probs.requires_grad

        if not self.parameters.on_policy:
            importance_sampling = (
                torch.exp(log_probs.detach()) / training_input.tgt_out_probs
            )
        else:
            importance_sampling = 1
        # add negative sign because we take gradient descent but we want to
        # maximize rewards
        batch_loss = -importance_sampling * log_probs * (reward - b)
        # rl_loss = - 1. / batch_size * torch.sum(log_probs * (reward - b))
        rl_loss = 1.0 / batch_size * torch.sum(batch_loss)
        rl_loss.backward()
        self.rl_opt.step()
        self.rl_opt.zero_grad()
        rl_loss = rl_loss.detach().cpu().numpy()
        baseline_loss = baseline_loss.detach().cpu().numpy()

        advantage = reward - b
        log_probs = log_probs.detach()

        self.minibatch += 1
        t2 = time.time()
        logger.info(
            "{} batch: rl_loss={}, baseline_loss={}, time={}".format(
                self.minibatch, rl_loss, baseline_loss, t2 - t1
            )
        )

        return (log_probs, advantage, rl_loss, baseline_loss)
