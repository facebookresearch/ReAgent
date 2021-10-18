#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import List, Optional, Tuple

import reagent.core.types as rlt
import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.core.dataclasses import field
from reagent.core.parameters import Seq2SlateParameters
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.model_utils.seq2slate_utils import Seq2SlateMode
from reagent.models.seq2slate import BaselineNet, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.training.ranking.helper import ips_clamp
from reagent.training.reagent_lightning_module import ReAgentLightningModule


logger = logging.getLogger(__name__)


class Seq2SlateTrainer(ReAgentLightningModule):
    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        params: Seq2SlateParameters = field(  # noqa: B008
            default_factory=Seq2SlateParameters
        ),
        baseline_net: Optional[BaselineNet] = None,
        baseline_warmup_num_batches: int = 0,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        baseline_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        policy_gradient_interval: int = 1,
        print_interval: int = 100,
        calc_cpe: bool = False,
        reward_network: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.seq2slate_net = seq2slate_net
        self.params = params
        self.policy_gradient_interval = policy_gradient_interval
        self.print_interval = print_interval

        self.baseline_net = baseline_net
        self.baseline_warmup_num_batches = baseline_warmup_num_batches

        self.rl_opt = policy_optimizer
        if self.baseline_net:
            self.baseline_opt = baseline_optimizer

        # use manual optimization to get more flexibility
        self.automatic_optimization = False

        assert not calc_cpe or reward_network is not None
        self.calc_cpe = calc_cpe
        self.reward_network = reward_network

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.rl_opt.make_optimizer_scheduler(self.seq2slate_net.parameters())
        )
        if self.baseline_net:
            optimizers.append(
                self.baseline_opt.make_optimizer_scheduler(
                    self.baseline_net.parameters()
                )
            )
        return optimizers

    def _compute_impt_smpl(
        self, model_propensities, logged_propensities
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logged_propensities = logged_propensities.reshape(-1, 1)
        assert (
            model_propensities.shape == logged_propensities.shape
            and len(model_propensities.shape) == 2
            and model_propensities.shape[1] == 1
        ), f"{model_propensities.shape} {logged_propensities.shape}"

        impt_smpl = model_propensities / logged_propensities
        clamped_impt_smpl = ips_clamp(impt_smpl, self.params.ips_clamp)
        return impt_smpl, clamped_impt_smpl

    # pyre-fixme [14]: overrides method defined in `ReAgentLightningModule` inconsistently
    def training_step(self, batch: rlt.PreprocessedRankingInput, batch_idx: int):
        assert type(batch) is rlt.PreprocessedRankingInput

        batch_size = batch.state.float_features.shape[0]

        reward = batch.slate_reward
        assert reward is not None

        optimizers = self.optimizers()
        if self.baseline_net:
            assert len(optimizers) == 2
            baseline_opt = optimizers[1]
        else:
            assert len(optimizers) == 1
        rl_opt = optimizers[0]

        if self.baseline_net:
            # Train baseline
            b = self.baseline_net(batch)
            baseline_loss = 1.0 / batch_size * torch.sum((b - reward) ** 2)
            baseline_opt.zero_grad()
            self.manual_backward(baseline_loss)
            baseline_opt.step()
        else:
            b = torch.zeros_like(reward)
            baseline_loss = torch.zeros(1)

        # Train Seq2Slate using REINFORCE
        # log probs of tgt seqs
        model_propensities = torch.exp(
            self.seq2slate_net(
                batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
            ).log_probs
        )
        b = b.detach()
        assert (
            b.shape == reward.shape == model_propensities.shape
        ), f"{b.shape} {reward.shape} {model_propensities.shape}"

        impt_smpl, clamped_impt_smpl = self._compute_impt_smpl(
            model_propensities, batch.tgt_out_probs
        )
        assert (
            impt_smpl.shape == clamped_impt_smpl.shape == reward.shape
        ), f"{impt_smpl.shape} {clamped_impt_smpl.shape} {reward.shape}"
        # gradient is only w.r.t model_propensities
        assert (
            not reward.requires_grad
            # pyre-fixme[16]: `Optional` has no attribute `requires_grad`.
            and not batch.tgt_out_probs.requires_grad
            and impt_smpl.requires_grad
            and clamped_impt_smpl.requires_grad
            and not b.requires_grad
        )
        # add negative sign because we take gradient descent but we want to
        # maximize rewards
        batch_obj_loss = -clamped_impt_smpl * (reward - b)
        obj_loss = torch.mean(batch_obj_loss)

        # condition to perform policy gradient update:
        # 1. no baseline
        # 2. or baseline is present and it passes the warm up stage
        # 3. the last policy gradient was performed policy_gradient_interval minibatches ago
        if (
            self.baseline_net is None
            or (self.all_batches_processed + 1) >= self.baseline_warmup_num_batches
        ):
            self.manual_backward(obj_loss)
            if (self.all_batches_processed + 1) % self.policy_gradient_interval == 0:
                rl_opt.step()
                rl_opt.zero_grad()
        else:
            logger.info("Not update RL model because now is baseline warmup phase")

        ips_loss = torch.mean(-impt_smpl * reward).cpu().detach().numpy()
        clamped_ips_loss = (
            torch.mean(-clamped_impt_smpl * reward).cpu().detach().numpy()
        )
        baseline_loss = baseline_loss.detach().cpu().numpy().item()
        advantage = (reward - b).detach().cpu().numpy()
        logged_slate_rank_probs = model_propensities.detach().cpu().numpy()

        if (self.all_batches_processed + 1) % self.print_interval == 0:
            logger.info(
                "{} batch: ips_loss={}, clamped_ips_loss={}, baseline_loss={}, max_ips={}, mean_ips={}, grad_update={}".format(
                    self.all_batches_processed + 1,
                    ips_loss,
                    clamped_ips_loss,
                    baseline_loss,
                    torch.max(impt_smpl),
                    torch.mean(impt_smpl),
                    (self.all_batches_processed + 1) % self.policy_gradient_interval
                    == 0,
                )
            )
        self.reporter.log(
            train_ips_score=torch.tensor(ips_loss).reshape(1),
            train_clamped_ips_score=torch.tensor(clamped_ips_loss).reshape(1),
            train_baseline_loss=torch.tensor(baseline_loss).reshape(1),
            train_logged_slate_rank_probs=torch.FloatTensor(logged_slate_rank_probs),
            train_ips_ratio=impt_smpl,
            train_clamped_ips_ratio=clamped_impt_smpl,
            train_advantages=advantage,
        )

    # pyre-ignore inconsistent override because lightning doesn't use types
    def validation_step(self, batch: rlt.PreprocessedRankingInput, batch_idx: int):
        seq2slate_net = self.seq2slate_net

        assert seq2slate_net.training is False

        logged_slate_rank_prob = torch.exp(
            seq2slate_net(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE)
            .log_probs.detach()
            .flatten()
            .cpu()
        )

        eval_baseline_loss = torch.tensor([0.0]).reshape(1)
        if self.baseline_net:
            baseline_net = self.baseline_net
            b = baseline_net(batch).detach()
            eval_baseline_loss = F.mse_loss(b, batch.slate_reward).cpu().reshape(1)
        else:
            b = torch.zeros_like(batch.slate_reward)

        eval_advantage = (
            # pyre-fixme[58]: `-` is not supported for operand types
            #  `Optional[torch.Tensor]` and `Any`.
            (batch.slate_reward - b)
            .flatten()
            .cpu()
        )

        ranked_slate_output = seq2slate_net(batch, Seq2SlateMode.RANK_MODE, greedy=True)
        ranked_slate_rank_prob = ranked_slate_output.ranked_per_seq_probs.cpu()

        self.reporter.log(
            eval_baseline_loss=eval_baseline_loss,
            eval_advantages=eval_advantage,
            logged_slate_rank_probs=logged_slate_rank_prob,
            ranked_slate_rank_probs=ranked_slate_rank_prob,
        )

        if not self.calc_cpe:
            return

        edp_g = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net,
            # pyre-fixme[6]: Expected `Module` for 2nd param but got
            #  `Optional[nn.Module]`.
            self.reward_network,
            batch,
            eval_greedy=True,
        )

        edp_ng = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net,
            # pyre-fixme[6]: Expected `Module` for 2nd param but got
            #  `Optional[nn.Module]`.
            self.reward_network,
            batch,
            eval_greedy=False,
        )

        return edp_g, edp_ng

    # pyre-fixme[14]: Inconsistent override
    def validation_epoch_end(
        self, outputs: Optional[List[Tuple[EvaluationDataPage, EvaluationDataPage]]]
    ):
        if self.calc_cpe:
            assert outputs is not None
            eval_data_pages_g, eval_data_pages_ng = None, None
            for edp_g, edp_ng in outputs:
                if eval_data_pages_g is None and eval_data_pages_ng is None:
                    eval_data_pages_g = edp_g
                    eval_data_pages_ng = edp_ng
                else:
                    # pyre-fixme[16]: `Optional` has no attribute `append`
                    eval_data_pages_g.append(edp_g)
                    eval_data_pages_ng.append(edp_ng)
            self.reporter.log(
                eval_data_pages_g=eval_data_pages_g,
                eval_data_pages_ng=eval_data_pages_ng,
            )
