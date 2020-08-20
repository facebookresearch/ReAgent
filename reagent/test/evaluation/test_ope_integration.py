import logging
import unittest

import numpy as np
import torch
from reagent.core import types as rlt
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.ope_adapter import OPEstimatorAdapter
from reagent.ope.estimators.contextual_bandits_estimators import (
    DMEstimator,
    DoublyRobustEstimator,
    IPSEstimator,
    SwitchDREstimator,
    SwitchEstimator,
)
from reagent.test.evaluation.test_evaluation_data_page import (
    FakeSeq2SlateRewardNetwork,
    FakeSeq2SlateTransformerNet,
)


logger = logging.getLogger(__name__)


class TestOPEModuleAlgs(unittest.TestCase):
    GAMMA = 0.9
    CPE_PASS_BAR = 1.0
    CPE_MAX_VALUE = 2.0
    MAX_HORIZON = 1000
    NOISE_EPSILON = 0.3
    EPISODES = 2

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
        predicted propensities: 0.4, 0.3, 0.7

        When eval_greedy=True:

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


        When eval_greedy=False:

        Only Inverse Propensity Scores would be accurate. Because it would be too
        expensive to compute all possible slates' propensities and predicted rewards
        for Direct Method.

        The expected IPS = (0.4 / 0.2 * 4 + 0.3 / 0.5 * 5 + 0.7 / 0.4 * 7) / 3
        """
        batch_size = 3
        state_dim = 3
        src_seq_len = 2
        tgt_seq_len = 2
        candidate_dim = 2

        reward_net = FakeSeq2SlateRewardNetwork()
        seq2slate_net = FakeSeq2SlateTransformerNet()

        src_seq = torch.eye(candidate_dim).repeat(batch_size, 1, 1)
        tgt_out_idx = torch.LongTensor([[3, 2], [3, 2], [3, 2]])
        tgt_out_seq = src_seq[
            torch.arange(batch_size).repeat_interleave(tgt_seq_len),
            tgt_out_idx.flatten() - 2,
        ].reshape(batch_size, tgt_seq_len, candidate_dim)

        ptb = rlt.PreprocessedTrainingBatch(
            training_input=rlt.PreprocessedRankingInput(
                state=rlt.FeatureData(float_features=torch.eye(state_dim)),
                src_seq=rlt.FeatureData(float_features=src_seq),
                tgt_out_seq=rlt.FeatureData(float_features=tgt_out_seq),
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

        edp = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net, reward_net, ptb.training_input, eval_greedy=True
        )
        logger.info("---------- Start evaluating eval_greedy=True -----------------")
        doubly_robust_estimator = OPEstimatorAdapter(DoublyRobustEstimator())
        dm_estimator = OPEstimatorAdapter(DMEstimator())
        ips_estimator = OPEstimatorAdapter(IPSEstimator())
        switch_estimator = OPEstimatorAdapter(SwitchEstimator())
        switch_dr_estimator = OPEstimatorAdapter(SwitchDREstimator())

        doubly_robust = doubly_robust_estimator.estimate(edp)
        inverse_propensity = ips_estimator.estimate(edp)
        direct_method = dm_estimator.estimate(edp)

        # Verify that Switch with low exponent is equivalent to IPS
        switch_ips = switch_estimator.estimate(edp, exp_base=1)
        # Verify that Switch with no candidates is equivalent to DM
        switch_dm = switch_estimator.estimate(edp, candidates=0)
        # Verify that SwitchDR with low exponent is equivalent to DR
        switch_dr_dr = switch_dr_estimator.estimate(edp, exp_base=1)
        # Verify that SwitchDR with no candidates is equivalent to DM
        switch_dr_dm = switch_dr_estimator.estimate(edp, candidates=0)

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
        self.assertAlmostEqual(switch_ips.raw, inverse_propensity.raw, delta=1e-6)
        self.assertAlmostEqual(switch_dm.raw, direct_method.raw, delta=1e-6)
        self.assertAlmostEqual(switch_dr_dr.raw, doubly_robust.raw, delta=1e-6)
        self.assertAlmostEqual(switch_dr_dm.raw, direct_method.raw, delta=1e-6)
        logger.info("---------- Finish evaluating eval_greedy=True -----------------")

        logger.info("---------- Start evaluating eval_greedy=False -----------------")
        edp = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net, reward_net, ptb.training_input, eval_greedy=False
        )
        doubly_robust_estimator = OPEstimatorAdapter(DoublyRobustEstimator())
        dm_estimator = OPEstimatorAdapter(DMEstimator())
        ips_estimator = OPEstimatorAdapter(IPSEstimator())

        doubly_robust = doubly_robust_estimator.estimate(edp)
        inverse_propensity = ips_estimator.estimate(edp)
        direct_method = dm_estimator.estimate(edp)
        self.assertAlmostEqual(
            inverse_propensity.raw,
            (0.4 / 0.2 * 4 + 0.3 / 0.5 * 5 + 0.7 / 0.4 * 7) / 3,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            inverse_propensity.normalized,
            inverse_propensity.raw / avg_logged_reward,
            delta=1e-6,
        )
        logger.info("---------- Finish evaluating eval_greedy=False -----------------")
