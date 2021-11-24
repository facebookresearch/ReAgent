#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import random
import unittest

import numpy as np
import torch
from reagent.ope.estimators.contextual_bandits_estimators import (
    Action,
    ActionDistribution,
    ActionSpace,
    BanditsEstimatorInput,
    DMEstimator,
    DoublyRobustEstimator,
    IPSEstimator,
    LogSample,
    ModelOutputs,
    SwitchDREstimator,
    SwitchEstimator,
)


class TestSwitchEstimators(unittest.TestCase):
    """
    These unit tests verify basic properties of the Switch estimators, in that
    when the threshold is low, the model-based DM estimator is used and when the
    threshold is high, the propensity score estimator is used.
    """

    NUM_ACTIONS = 2
    DR_EPSILON = 0.05

    def setUp(self) -> None:
        random.seed(0)
        torch.random.manual_seed(0)
        np.random.seed(0)
        self.action_space = ActionSpace(TestSwitchEstimators.NUM_ACTIONS)
        self.sample1 = LogSample(
            context=0,
            log_action=Action(0),
            log_reward=1.0,
            log_action_probabilities=ActionDistribution(torch.tensor([0.7, 0.3])),
            tgt_action_probabilities=ActionDistribution([0.6, 0.4]),
            tgt_action=Action(1),
            model_outputs=ModelOutputs(0.5, [0.4, 0.5]),
        )
        self.sample2 = LogSample(
            context=0,
            log_action=Action(1),
            log_reward=0.0,
            log_action_probabilities=ActionDistribution([0.5, 0.5]),
            tgt_action_probabilities=ActionDistribution([0.7, 0.3]),
            tgt_action=Action(0),
            model_outputs=ModelOutputs(0.0, [0.0, 0.0]),
        )
        self.bandit_input = BanditsEstimatorInput(
            self.action_space, [self.sample1, self.sample2], True
        )
        SwitchEstimator.EXP_BASE = 1.5
        SwitchEstimator.CANDIDATES = 21

    def test_switch_equal_to_ips(self):
        """
        Switch with tau set at the max value should be equal to IPS
        """
        # Setting the base to 1 will cause all candidates to be the maximum threshold
        SwitchEstimator.EXP_BASE = 1
        switch = SwitchEstimator(rmax=1.0).evaluate(self.bandit_input)
        ips = IPSEstimator().evaluate(self.bandit_input)
        self.assertAlmostEqual(ips.estimated_reward, switch.estimated_reward)

    def test_switch_dr_equal_to_dr(self):
        """
        Switch-DR with tau set at the max value should be equal to DR
        """
        # Setting the base to 1 will cause all candidates to be the maximum threshold
        SwitchEstimator.EXP_BASE = 1
        switch = SwitchDREstimator(rmax=1.0).evaluate(self.bandit_input)
        dr = DoublyRobustEstimator().evaluate(self.bandit_input)
        self.assertAlmostEqual(
            dr.estimated_reward,
            switch.estimated_reward,
            delta=TestSwitchEstimators.DR_EPSILON,
        )

    def test_switch_equal_to_dm(self):
        """
        Switch with tau set at the min value should be equal to DM
        """
        # Setting candidates to 0 will default to tau being the minimum threshold
        SwitchEstimator.CANDIDATES = 0
        switch = SwitchEstimator(rmax=1.0).evaluate(self.bandit_input)
        dm = DMEstimator().evaluate(self.bandit_input)
        self.assertAlmostEqual(dm.estimated_reward, switch.estimated_reward)

    def test_switch_dr_equal_to_dm(self):
        """
        Switch-DR with tau set at the min value should be equal to DM
        """
        # Setting candidates to 0 will default to tau being the minimum threshold
        SwitchEstimator.CANDIDATES = 0
        switch = SwitchDREstimator(rmax=1.0).evaluate(self.bandit_input)
        dm = DMEstimator().evaluate(self.bandit_input)
        self.assertAlmostEqual(dm.estimated_reward, switch.estimated_reward)
