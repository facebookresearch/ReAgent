#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Optional, Tuple

import numpy as np
import reagent.types as rlt
import torch
from reagent.core.dataclasses import field
from reagent.core.tracker import observable
from reagent.models.seq2slate import BaselineNet, Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import Seq2SlateParameters
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


@observable(
    train_ips_score=torch.Tensor,
    train_clamped_ips_score=torch.Tensor,
    train_baseline_loss=torch.Tensor,
    train_log_probs=torch.Tensor,
    train_ips_ratio=torch.Tensor,
    train_clamped_ips_ratio=torch.Tensor,
    train_advantages=torch.Tensor,
)
class Seq2SlateTrainer(Trainer):
    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        minibatch_size: int = 1024,
        parameters: Seq2SlateParameters = field(  # noqa: B008
            default_factory=Seq2SlateParameters
        ),
        baseline_net: Optional[BaselineNet] = None,
        baseline_warmup_num_batches: int = 0,
        use_gpu: bool = False,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        baseline_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        print_interval: int = 100,
    ) -> None:
        self.seq2slate_net = seq2slate_net
        self.parameters = parameters
        self.use_gpu = use_gpu
        self.print_interval = print_interval

        self.minibatch_size = minibatch_size
        self.minibatch = 0

        self.baseline_net = baseline_net
        self.baseline_warmup_num_batches = baseline_warmup_num_batches

        self.rl_opt = policy_optimizer.make_optimizer(self.seq2slate_net.parameters())
        if self.baseline_net:
            self.baseline_opt = baseline_optimizer.make_optimizer(
                # pyre-fixme[16]: `Optional` has no attribute `parameters`.
                self.baseline_net.parameters()
            )

    def warm_start_components(self):
        components = ["seq2slate_net"]
        if self.baseline_net:
            components.append("baseline_net")
        return components

    def _compute_impt_smpl(
        self, model_propensities, logged_propensities
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logged_propensities = logged_propensities.reshape(-1, 1)
        assert (
            model_propensities.shape == logged_propensities.shape
            and len(model_propensities.shape) == 2
            and model_propensities.shape[1] == 1
        ), f"{model_propensities.shape} {logged_propensities.shape}"

        clamped_impt_smpl = impt_smpl = model_propensities / logged_propensities
        if self.parameters.importance_sampling_clamp_max is not None:
            clamped_impt_smpl = torch.clamp(
                impt_smpl, 0, self.parameters.importance_sampling_clamp_max
            )
        return impt_smpl, clamped_impt_smpl

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

        impt_smpl, clamped_impt_smpl = self._compute_impt_smpl(
            torch.exp(log_probs.detach()), training_input.tgt_out_probs
        )
        assert (
            impt_smpl.shape == clamped_impt_smpl.shape == reward.shape
        ), f"{impt_smpl.shape} {clamped_impt_smpl.shape} {reward.shape}"
        # gradient is only w.r.t log_probs
        assert (
            not reward.requires_grad
            and not impt_smpl.requires_grad
            and not clamped_impt_smpl.requires_grad
            and not b.requires_grad
            and log_probs.requires_grad
        )

        # add negative sign because we take gradient descent but we want to
        # maximize rewards
        batch_loss = -log_probs * (reward - b)
        if not self.parameters.on_policy:
            batch_loss *= clamped_impt_smpl
        rl_loss = torch.mean(batch_loss)

        if (
            self.baseline_net is None
            or self.minibatch >= self.baseline_warmup_num_batches
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
        ips_rl_loss = torch.mean(-impt_smpl * reward).cpu().numpy()
        clamped_ips_rl_loss = torch.mean(-clamped_impt_smpl * reward).cpu().numpy()
        baseline_loss = baseline_loss.detach().cpu().numpy().item()

        advantage = (reward - b).detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()

        self.minibatch += 1
        if self.minibatch % self.print_interval == 0:
            logger.info(
                "{} batch: obj_rl_loss={}, ips_rl_loss={}, baseline_loss={}, max_ips={}, mean_ips={}, clamp={}".format(
                    self.minibatch,
                    obj_rl_loss,
                    ips_rl_loss,
                    baseline_loss,
                    torch.max(impt_smpl),
                    torch.mean(impt_smpl),
                    self.parameters.importance_sampling_clamp_max,
                )
            )
        # See RankingTrainingPageHandler.finish() function in page_handler.py
        # pyre-fixme[16]: `Seq2SlateTrainer` has no attribute
        #  `notify_observers`.
        self.notify_observers(
            train_ips_score=torch.tensor(ips_rl_loss).reshape(1),
            train_clamped_ips_score=torch.tensor(clamped_ips_rl_loss).reshape(1),
            train_baseline_loss=torch.tensor(baseline_loss).reshape(1),
            train_log_probs=torch.FloatTensor(log_probs),
            train_ips_ratio=impt_smpl,
            train_clamped_ips_ratio=clamped_impt_smpl,
            train_advantages=advantage,
        )
