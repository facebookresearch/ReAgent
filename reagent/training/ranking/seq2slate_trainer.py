#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Optional

import numpy as np
import reagent.types as rlt
import torch
from reagent.models.seq2slate import BaselineNet, Seq2SlateMode, Seq2SlateTransformerNet
from reagent.parameters import Seq2SlateTransformerParameters
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class Seq2SlateTrainer(Trainer):
    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        parameters: Seq2SlateTransformerParameters,
        minibatch_size: int,
        baseline_net: Optional[BaselineNet] = None,
        use_gpu: bool = False,
    ) -> None:
        self.parameters = parameters
        self.use_gpu = use_gpu
        self.seq2slate_net = seq2slate_net
        self.baseline_net = baseline_net
        self.minibatch_size = minibatch_size
        self.minibatch = 0
        self.rl_opt = torch.optim.Adam(
            self.seq2slate_net.parameters(),
            lr=self.parameters.transformer.learning_rate,
            amsgrad=True,
        )
        if self.baseline_net:
            assert self.parameters.baseline
            self.baseline_opt = torch.optim.Adam(
                # pyre-fixme[16]: `Optional` has no attribute `parameters`.
                self.baseline_net.parameters(),
                # pyre-fixme[16]: `Optional` has no attribute `learning_rate`.
                lr=self.parameters.baseline.learning_rate,
                amsgrad=True,
            )
        assert (
            self.parameters.importance_sampling_clamp_max is None
            or not self.parameters.on_policy
        ), (
            "importance_sampling_clamp_max is not useful and should "
            "be set to None in on-policy learning"
        )

    def warm_start_components(self):
        components = ["seq2slate_net"]
        if self.baseline_net:
            components.append("baseline_net")
        return components

    def _compute_impt_sampling(
        self, model_propensities, logged_propensities
    ) -> torch.Tensor:
        device = model_propensities.device
        batch_size = model_propensities.shape[0]
        if not self.parameters.on_policy:
            return model_propensities / logged_propensities
        # on policy performs no importance sampling correction = setting IS to 1
        return torch.ones(batch_size, 1, device=device)

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        assert type(training_batch) is rlt.PreprocessedTrainingBatch
        training_input = training_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedRankingInput)

        batch_size = training_input.state.float_features.shape[0]
        device = torch.device("cuda") if self.use_gpu else torch.device("cpu")

        reward = training_input.slate_reward
        batch_size = training_input.state.float_features.shape[0]
        assert reward is not None

        if self.baseline_net:
            # Train baseline
            # pyre-fixme[29]: `Optional[BaselineNet]` is not a function.
            b = self.baseline_net(training_input)
            baseline_loss = 1.0 / batch_size * torch.sum((b - reward) ** 2)
            self.baseline_opt.zero_grad()
            baseline_loss.backward()
            self.baseline_opt.step()
        else:
            b = torch.zeros_like(reward, device=device)
            baseline_loss = torch.zeros(1, device=device)

        # Train Seq2Slate using REINFORCE
        # log probs of tgt seqs
        log_probs = self.seq2slate_net(
            training_input, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
        ).log_probs
        b = b.detach()
        assert (
            b.shape == reward.shape == log_probs.shape
        ), f"{b.shape} {reward.shape} {log_probs.shape}"

        importance_sampling = self._compute_impt_sampling(
            torch.exp(log_probs.detach()), training_input.tgt_out_probs
        )
        clamped_importance_sampling = importance_sampling
        if self.parameters.importance_sampling_clamp_max is not None:
            clamped_importance_sampling = torch.clamp(
                importance_sampling, 0, self.parameters.importance_sampling_clamp_max
            )

        assert importance_sampling.shape == reward.shape

        # gradient is only w.r.t log_probs
        assert (
            not reward.requires_grad
            and not importance_sampling.requires_grad
            and not clamped_importance_sampling.requires_grad
            and not b.requires_grad
            and log_probs.requires_grad
        )

        # add negative sign because we take gradient descent but we want to
        # maximize rewards
        batch_loss = -clamped_importance_sampling * log_probs * (reward - b)
        rl_loss = 1.0 / batch_size * torch.sum(batch_loss)

        if (
            self.parameters.baseline is None
            # pyre-fixme[16]: `Optional` has no attribute `warmup_num_batches`.
            or self.minibatch >= self.parameters.baseline.warmup_num_batches
        ):
            self.rl_opt.zero_grad()
            rl_loss.backward()
            self.rl_opt.step()
        else:
            logger.info("Not update RL model because now is baseline warmup phase")

        # obj_rl_loss is the objective we take gradient with regard to
        # ips_rl_loss is the sum of importance sampling weighted rewards, which gives
        # the same gradient when we don't use baseline or clamp.
        # obj_rl_loss is used to get gradient becaue it is in the logarithmic form
        # thus more stable.
        # ips_rl_loss is more useful as an offline evaluation metric
        obj_rl_loss = rl_loss.detach().cpu().numpy()
        ips_rl_loss = (
            (-1.0 / batch_size * torch.sum(importance_sampling * reward)).cpu().numpy()
        )
        baseline_loss = baseline_loss.detach().cpu().numpy().item()

        advantage = (reward - b).detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()

        self.minibatch += 1
        if self.minibatch % 10 == 0:
            logger.info(
                "{} batch: obj_rl_loss={}, ips_rl_loss={}, baseline_loss={}, max_ips={}, mean_ips={}, clamp={}".format(
                    self.minibatch,
                    obj_rl_loss,
                    ips_rl_loss,
                    baseline_loss,
                    torch.max(importance_sampling),
                    torch.mean(importance_sampling),
                    self.parameters.importance_sampling_clamp_max,
                )
            )

        return {
            "per_seq_probs": np.exp(log_probs),
            "advantage": advantage,
            "obj_rl_loss": obj_rl_loss,
            "ips_rl_loss": ips_rl_loss,
            "baseline_loss": baseline_loss,
        }
