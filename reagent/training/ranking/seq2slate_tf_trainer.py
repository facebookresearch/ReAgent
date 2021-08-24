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
from reagent.models.seq2slate import Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule


logger = logging.getLogger(__name__)


class Seq2SlateTeacherForcingTrainer(ReAgentLightningModule):
    """
    Seq2Slate learned in a teach-forcing fashion (only used if the
    the ground-truth sequences are available)
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        params: Seq2SlateParameters,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        policy_gradient_interval: int = 1,
        print_interval: int = 100,
        calc_cpe: bool = False,
        reward_network: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.params = params
        self.policy_gradient_interval = policy_gradient_interval
        self.print_interval = print_interval
        self.seq2slate_net = seq2slate_net
        self.policy_optimizer = policy_optimizer
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

        # use manual optimization to get more flexibility
        self.automatic_optimization = False

        assert not calc_cpe or reward_network is not None
        self.calc_cpe = calc_cpe
        self.reward_network = reward_network

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.policy_optimizer.make_optimizer_scheduler(
                self.seq2slate_net.parameters()
            )
        )
        return optimizers

    # pyre-fixme [14]: overrides method defined in `ReAgentLightningModule` inconsistently
    def training_step(self, batch: rlt.PreprocessedRankingInput, batch_idx: int):
        assert type(batch) is rlt.PreprocessedRankingInput

        log_probs = self.seq2slate_net(
            batch, mode=Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE
        ).log_probs
        assert log_probs.requires_grad

        assert batch.optim_tgt_out_idx is not None
        labels = self._transform_label(batch.optim_tgt_out_idx)
        assert not labels.requires_grad
        loss = self.kl_div_loss(log_probs, labels)

        self.manual_backward(loss)
        if (self.all_batches_processed + 1) % self.policy_gradient_interval == 0:
            opt = self.optimizers()[0]
            opt.step()
            opt.zero_grad()

        loss = loss.detach().cpu().numpy()
        log_probs = log_probs.detach()
        if (self.all_batches_processed + 1) % self.print_interval == 0:
            logger.info(f"{self.all_batches_processed + 1} batch: loss={loss}")

        return log_probs, loss

    def _transform_label(self, optim_tgt_out_idx: torch.Tensor):
        label_size = self.seq2slate_net.max_src_seq_len + 2
        label = F.one_hot(optim_tgt_out_idx, label_size)
        return label.float()

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

        ranked_slate_output = seq2slate_net(batch, Seq2SlateMode.RANK_MODE, greedy=True)
        ranked_slate_rank_prob = ranked_slate_output.ranked_per_seq_probs.cpu()

        self.reporter.log(
            logged_slate_rank_probs=logged_slate_rank_prob,
            ranked_slate_rank_probs=ranked_slate_rank_prob,
        )

        if not self.calc_cpe:
            return

        edp_g = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net,
            self.reward_network,
            batch,
            eval_greedy=True,
        )

        edp_ng = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net,
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
