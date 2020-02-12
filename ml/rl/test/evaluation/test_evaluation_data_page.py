#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import numpy as np
import torch
import torch.nn as nn
from ml.rl import types as rlt
from ml.rl.evaluation.doubly_robust_estimator import DoublyRobustEstimator
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.models.seq2slate import Seq2SlateMode
from ml.rl.parameters import (
    BaselineParameters,
    Seq2SlateTransformerParameters,
    TransformerParameters,
)
from ml.rl.training.ranking.seq2slate_trainer import Seq2SlateTrainer


logger = logging.getLogger(__name__)


class FakeSeq2SlateRewardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_parms = nn.Linear(1, 1)

    def forward(
        self,
        state: torch.Tensor,
        src_seq: torch.Tensor,
        tgt_out_seq: torch.Tensor,
        src_src_mask: torch.Tensor,
        slate_reward: torch.Tensor,
        tgt_out_idx: torch.Tensor,
    ):
        batch_size = state.shape[0]
        rewards = []
        for i in range(batch_size):
            rewards.append(self._forward(state[i], tgt_out_idx[i]))
        return torch.tensor(rewards).float()

    def _forward(self, state: torch.Tensor, tgt_out_idx: torch.Tensor):
        if (state == torch.tensor([1.0, 0.0, 0.0])).all():
            if (tgt_out_idx == torch.tensor([2, 3])).all():
                return 1.0
            else:
                return 2.0
        elif (state == torch.tensor([0.0, 1.0, 0.0])).all():
            if (tgt_out_idx == torch.tensor([2, 3])).all():
                return 3.0
            else:
                return 4.0
        elif (state == torch.tensor([0.0, 0.0, 1.0])).all():
            if (tgt_out_idx == torch.tensor([2, 3])).all():
                return 5.0
            else:
                return 6.0


class FakeSeq2SlateTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_parms = nn.Linear(1, 1)

    def forward(self, input: rlt.PreprocessedRankingInput, mode: str, greedy: bool):
        # The creation of evaluation data pages only uses these specific arguments
        assert greedy and mode == Seq2SlateMode.RANK_MODE
        batch_size = input.state.float_features.shape[0]
        ranked_tgt_out_idx = []

        for i in range(batch_size):
            ranked_tgt_out_idx.append(self._forward(input.state.float_features[i]))

        return rlt.RankingOutput(
            ranked_tgt_out_idx=torch.tensor(ranked_tgt_out_idx).long()
        )

    def _forward(self, state: torch.Tensor):
        if (state == torch.tensor([1.0, 0.0, 0.0])).all():
            return [2, 3]
        elif (state == torch.tensor([0.0, 1.0, 0.0])).all():
            return [3, 2]
        elif (state == torch.tensor([0.0, 0.0, 1.0])).all():
            return [2, 3]


class TestEvaluationDataPage(unittest.TestCase):
    def test_seq2slate_eval_data_page(self):
        """
        Create 3 slate ranking logs and evaluate using Direct Method, Inverse
        Propensity Scores, and Doubly Robust.

        The logs are as follows:
        state: [1, 0, 0], [0, 1, 0], [0, 0, 1]
        indices in logged slates: [3, 2], [3, 2], [3, 2]
        model output indices: [2, 3], [3, 2], [2, 3]
        logged reward: 4, 5, 7
        logged propensities: 0.2, 0.5, 0.4
        predicted rewards on logged slates: 2, 4, 6
        predicted rewards on model outputted slates: 1, 4, 5

        Direct Method uses the predicted rewards on model outputted slates.
        Thus the result is expected to be (1 + 4 + 5) / 3

        Inverse Propensity Scores would scale the reward by 1.0 / logged propensities
        whenever the model output slate matches with the logged slate.
        Since only the second log matches with the model output, the IPS result
        is expected to be 5 / 0.5 / 3

        Doubly Robust is the sum of the direct method result and propensity-scaled
        reward difference; the latter is defined as:
        1.0 / logged_propensities * (logged reward - predicted reward on logged slate)
         * Indicator(model slate == logged slate)
        Since only the second logged slate matches with the model outputted slate,
        the DR result is expected to be (1 + 4 + 5) / 3 + 1.0 / 0.5 * (5 - 4) / 3
        """
        batch_size = 3
        state_dim = 3
        src_seq_len = 2
        tgt_seq_len = 2
        candidate_dim = 2

        reward_net = FakeSeq2SlateRewardNetwork()
        seq2slate_net = FakeSeq2SlateTransformerNet()
        baseline_net = nn.Linear(1, 1)
        trainer = Seq2SlateTrainer(
            seq2slate_net,
            baseline_net,
            # these parameters are not used in this test but some of them
            # are required in __init__ function of Seq2SlateTrainer
            parameters=Seq2SlateTransformerParameters(
                transformer=TransformerParameters(
                    num_heads=0, dim_model=0, dim_feedforward=0, num_stacked_layers=0
                ),
                baseline=BaselineParameters(dim_feedforward=0, num_stacked_layers=0),
                on_policy=False,
            ),
            minibatch_size=3,
            use_gpu=False,
        )

        src_seq = torch.eye(candidate_dim).repeat(batch_size, 1, 1)
        tgt_out_idx = torch.LongTensor([[3, 2], [3, 2], [3, 2]])
        tgt_out_seq = src_seq[
            torch.arange(batch_size).repeat_interleave(tgt_seq_len),  # type: ignore
            tgt_out_idx.flatten() - 2,
        ].reshape(batch_size, tgt_seq_len, candidate_dim)

        ptb = rlt.PreprocessedTrainingBatch(
            training_input=rlt.PreprocessedRankingInput(
                state=rlt.PreprocessedFeatureVector(
                    float_features=torch.eye(state_dim)
                ),
                src_seq=rlt.PreprocessedFeatureVector(float_features=src_seq),
                tgt_out_seq=rlt.PreprocessedFeatureVector(float_features=tgt_out_seq),
                src_src_mask=torch.ones(batch_size, src_seq_len, src_seq_len),
                tgt_out_idx=tgt_out_idx,
                tgt_out_probs=torch.tensor([0.2, 0.5, 0.4]),
                slate_reward=torch.tensor([4.0, 5.0, 7.0]),
            ),
            extras=rlt.ExtraData(
                sequence_number=torch.tensor([0, 0, 0]),
                mdp_id=np.array(["0", "1", "2"]),
            ),
        )

        edp = EvaluationDataPage.create_from_training_batch(ptb, trainer, reward_net)
        doubly_robust_estimator = DoublyRobustEstimator()
        direct_method, inverse_propensity, doubly_robust = doubly_robust_estimator.estimate(
            edp
        )
        logger.info(f"{direct_method}, {inverse_propensity}, {doubly_robust}")

        avg_logged_reward = (4 + 5 + 7) / 3
        self.assertAlmostEqual(direct_method.raw, (1 + 4 + 5) / 3, delta=1e-6)
        self.assertAlmostEqual(
            direct_method.normalized, direct_method.raw / avg_logged_reward, delta=1e-6
        )
        self.assertAlmostEqual(inverse_propensity.raw, 5 / 0.5 / 3, delta=1e-6)
        self.assertAlmostEqual(
            inverse_propensity.normalized,
            inverse_propensity.raw / avg_logged_reward,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            doubly_robust.raw, direct_method.raw + 1 / 0.5 * (5 - 4) / 3, delta=1e-6
        )
        self.assertAlmostEqual(
            doubly_robust.normalized, doubly_robust.raw / avg_logged_reward, delta=1e-6
        )
