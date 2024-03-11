#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import random
import unittest

import numpy as np
import torch
from reagent.core import types as rlt
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.ope_adapter import (
    OPEstimatorAdapter,
    SequentialOPEstimatorAdapter,
)
from reagent.ope.estimators.contextual_bandits_estimators import (
    DMEstimator,
    DoublyRobustEstimator,
    IPSEstimator,
    SwitchDREstimator,
    SwitchEstimator,
)
from reagent.ope.estimators.sequential_estimators import (
    DoublyRobustEstimator as SeqDREstimator,
    EpsilonGreedyRLPolicy,
    RandomRLPolicy,
    RLEstimatorInput,
)
from reagent.ope.estimators.types import Action, ActionSpace
from reagent.ope.test.envs import PolicyLogGenerator
from reagent.ope.test.gridworld import GridWorld, NoiseGridWorldModel
from reagent.ope.trainers.rl_tabular_trainers import (
    DPTrainer,
    DPValueFunction,
    TabularPolicy,
)
from reagent.test.evaluation.test_evaluation_data_page import (
    FakeSeq2SlateRewardNetwork,
    FakeSeq2SlateTransformerNet,
)


logger = logging.getLogger(__name__)


def rlestimator_input_to_edp(
    input: RLEstimatorInput, num_actions: int
) -> EvaluationDataPage:
    mdp_ids = []
    logged_propensities = []
    logged_rewards = []
    action_mask = []
    model_propensities = []
    model_values = []

    for mdp in input.log:
        mdp_id = len(mdp_ids)
        for t in mdp:
            mdp_ids.append(mdp_id)
            logged_propensities.append(t.action_prob)
            logged_rewards.append(t.reward)
            assert t.action is not None
            action_mask.append(
                [1 if x == t.action.value else 0 for x in range(num_actions)]
            )
            assert t.last_state is not None
            model_propensities.append(
                [
                    input.target_policy(t.last_state)[Action(x)]
                    for x in range(num_actions)
                ]
            )
            assert input.value_function is not None
            model_values.append(
                [
                    input.value_function(t.last_state, Action(x))
                    for x in range(num_actions)
                ]
            )

    return EvaluationDataPage(
        mdp_id=torch.tensor(mdp_ids).reshape(len(mdp_ids), 1),
        logged_propensities=torch.tensor(logged_propensities).reshape(
            (len(logged_propensities), 1)
        ),
        logged_rewards=torch.tensor(logged_rewards).reshape((len(logged_rewards), 1)),
        action_mask=torch.tensor(action_mask),
        model_propensities=torch.tensor(model_propensities),
        model_values=torch.tensor(model_values),
        sequence_number=torch.tensor([]),
        model_rewards=torch.tensor([]),
        model_rewards_for_logged_action=torch.tensor([]),
    )


class TestOPEModuleAlgs(unittest.TestCase):
    GAMMA = 0.9
    CPE_PASS_BAR = 1.0
    CPE_MAX_VALUE = 2.0
    MAX_HORIZON = 1000
    NOISE_EPSILON = 0.3
    EPISODES = 2

    def test_gridworld_sequential_adapter(self):
        """
        Create a gridworld environment, logging policy, and target policy
        Evaluates target policy using the direct OPE sequential doubly robust estimator,
        then transforms the log into an evaluation data page which is passed to the ope adapter.

        This test is meant to verify the adaptation of EDPs into RLEstimatorInputs as employed
        by ReAgent since ReAgent provides EDPs to Evaluators. Going from EDP -> RLEstimatorInput
        is more involved than RLEstimatorInput -> EDP since the EDP does not store the state
        at each timestep in each MDP, only the corresponding logged outputs & model outputs.
        Thus, the adapter must do some tricks to represent these timesteps as states so the
        ope module can extract the correct outputs.

        Note that there is some randomness in the model outputs since the model is purposefully
        noisy. However, the same target policy is being evaluated on the same logged walks through
        the gridworld, so the two results should be close in value (within 1).

        """
        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)

        device = torch.device("cuda") if torch.cuda.is_available() else None

        gridworld = GridWorld.from_grid(
            [
                ["s", "0", "0", "0", "0"],
                ["0", "0", "0", "W", "0"],
                ["0", "0", "0", "0", "0"],
                ["0", "W", "0", "0", "0"],
                ["0", "0", "0", "0", "g"],
            ],
            max_horizon=TestOPEModuleAlgs.MAX_HORIZON,
        )

        action_space = ActionSpace(4)
        opt_policy = TabularPolicy(action_space)
        trainer = DPTrainer(gridworld, opt_policy)
        value_func = trainer.train(gamma=TestOPEModuleAlgs.GAMMA)

        behavivor_policy = RandomRLPolicy(action_space)
        target_policy = EpsilonGreedyRLPolicy(
            opt_policy, TestOPEModuleAlgs.NOISE_EPSILON
        )
        model = NoiseGridWorldModel(
            gridworld,
            action_space,
            epsilon=TestOPEModuleAlgs.NOISE_EPSILON,
            max_horizon=TestOPEModuleAlgs.MAX_HORIZON,
        )
        value_func = DPValueFunction(target_policy, model, TestOPEModuleAlgs.GAMMA)
        ground_truth = DPValueFunction(
            target_policy, gridworld, TestOPEModuleAlgs.GAMMA
        )

        log = []
        log_generator = PolicyLogGenerator(gridworld, behavivor_policy)
        num_episodes = TestOPEModuleAlgs.EPISODES
        for state in gridworld.states:
            for _ in range(num_episodes):
                log.append(log_generator.generate_log(state))

        estimator_input = RLEstimatorInput(
            gamma=TestOPEModuleAlgs.GAMMA,
            log=log,
            target_policy=target_policy,
            value_function=value_func,
            ground_truth=ground_truth,
        )

        edp = rlestimator_input_to_edp(estimator_input, len(model.action_space))

        dr_estimator = SeqDREstimator(
            weight_clamper=None, weighted=False, device=device
        )

        module_results = SequentialOPEstimatorAdapter.estimator_results_to_cpe_estimate(
            dr_estimator.evaluate(estimator_input)
        )
        adapter_results = SequentialOPEstimatorAdapter(
            dr_estimator, TestOPEModuleAlgs.GAMMA, device=device
        ).estimate(edp)

        self.assertAlmostEqual(
            adapter_results.raw,
            module_results.raw,
            delta=TestOPEModuleAlgs.CPE_PASS_BAR,
        ), f"OPE adapter results differed too much from underlying module (Diff: {abs(adapter_results.raw - module_results.raw)} > {TestOPEModuleAlgs.CPE_PASS_BAR})"
        self.assertLess(
            adapter_results.raw, TestOPEModuleAlgs.CPE_MAX_VALUE
        ), f"OPE adapter results are too large ({adapter_results.raw} > {TestOPEModuleAlgs.CPE_MAX_VALUE})"

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

        ptb = rlt.PreprocessedRankingInput(
            state=rlt.FeatureData(float_features=torch.eye(state_dim)),
            src_seq=rlt.FeatureData(float_features=src_seq),
            tgt_out_seq=rlt.FeatureData(float_features=tgt_out_seq),
            src_src_mask=torch.ones(batch_size, src_seq_len, src_seq_len),
            tgt_out_idx=tgt_out_idx,
            tgt_out_probs=torch.tensor([0.2, 0.5, 0.4]),
            slate_reward=torch.tensor([4.0, 5.0, 7.0]),
            extras=rlt.ExtraData(
                sequence_number=torch.tensor([0, 0, 0]),
                mdp_id=np.array(["0", "1", "2"]),
            ),
        )

        edp = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net, reward_net, ptb, eval_greedy=True
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
            seq2slate_net, reward_net, ptb, eval_greedy=False
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
