#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import unittest
from dataclasses import replace
from unittest.mock import MagicMock

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from reagent.core.types import CBInput
from reagent.evaluation.cb.policy_evaluator import PolicyEvaluator
from reagent.evaluation.cb.utils import add_importance_weights
from reagent.models.linear_regression import LinearRegressionUCB


def _compare_state_dicts(state_dict_1, state_dict_2):
    if len(state_dict_1) != len(state_dict_2):
        return False

    for (k_1, v_1), (k_2, v_2) in zip(
        sorted(state_dict_1.items()), sorted(state_dict_2.items())
    ):
        if k_1 != k_2:
            return False
        if not torch.allclose(v_1, v_2):
            return False
    return True


class TestPolicyEvaluator(unittest.TestCase):
    def setUp(self):
        self.policy_network = LinearRegressionUCB(2)
        self.eval_module = PolicyEvaluator(self.policy_network)
        self.batch = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 2],
                        [1, 3],
                    ],
                    [
                        [1, 4],
                        [1, 5],
                    ],
                ],
                dtype=torch.float,
            ),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[1.5], [2.3]], dtype=torch.float),
            action_log_probability=torch.tensor([[-2.0], [-3.0]], dtype=torch.float),
        )

    def test_process_all_data(self):

        state_dict_before = copy.deepcopy(self.eval_module.state_dict())
        self.eval_module._process_all_data(self.batch)
        state_dict_after = copy.deepcopy(self.eval_module.state_dict())

        # sum_weight_all_data_local got updated properly
        self.assertAlmostEqual(
            state_dict_after["sum_weight_all_data_local"].item()
            - state_dict_before["sum_weight_all_data_local"].item(),
            len(self.batch),
        )
        # sum_weight_all_data didn't change (bcs we haven't aggregated across instances yet)
        self.assertAlmostEqual(
            state_dict_after["sum_weight_all_data"].item(),
            state_dict_before["sum_weight_all_data"].item(),
        )

        # sum_reward_weighted_accepted_local got updated properly
        self.assertAlmostEqual(
            state_dict_after["sum_reward_weighted_accepted_local"].item(),
            state_dict_before["sum_reward_weighted_accepted_local"].item(),
        )
        # sum_reward_weighted_accepted didn't change (bcs we haven't aggregated across instances yet)
        self.assertAlmostEqual(
            state_dict_after["sum_reward_weighted_accepted"].item(),
            state_dict_before["sum_reward_weighted_accepted"].item(),
        )

        # sum_weight and sum_reward_weighted didn't change (as well as local values)
        self.assertAlmostEqual(
            state_dict_after["sum_weight_accepted"].item(),
            state_dict_before["sum_weight_accepted"].item(),
        )
        self.assertAlmostEqual(
            state_dict_after["sum_weight_accepted_local"].item(),
            state_dict_before["sum_weight_accepted_local"].item(),
        )
        self.assertAlmostEqual(
            state_dict_after["sum_reward_weighted_accepted"].item(),
            state_dict_before["sum_reward_weighted_accepted"].item(),
        )
        self.assertAlmostEqual(
            state_dict_after["sum_reward_weighted_accepted_local"].item(),
            state_dict_before["sum_reward_weighted_accepted_local"].item(),
        )

    def test_process_used_data_accept_some(self):
        # calling _process_used_data with non-zero weights should change the state and lead to correct reward value
        policy_network = LinearRegressionUCB(2)
        eval_module = PolicyEvaluator(policy_network)
        state_dict_before = copy.deepcopy(eval_module.state_dict())
        batch = add_importance_weights(
            self.batch, torch.tensor([[1], [1]], dtype=torch.long)
        )  # 2nd action matches, 1st doesn't
        importance_weight = torch.exp(-batch.action_log_probability[1, 0]).item()
        eval_module._process_used_data(batch)
        eval_module._aggregate_across_instances()
        state_dict_after = copy.deepcopy(eval_module.state_dict())
        self.assertFalse(_compare_state_dicts(state_dict_before, state_dict_after))
        self.assertEqual(eval_module.sum_weight_accepted_local.item(), 0.0)
        self.assertEqual(eval_module.sum_weight_accepted.item(), 1.0)
        self.assertEqual(
            eval_module.sum_importance_weight_accepted.item(), importance_weight
        )
        self.assertEqual(
            eval_module.sum_reward_weighted_accepted.item(),
            self.batch.reward[1, 0].item(),
        )
        self.assertAlmostEqual(
            eval_module.sum_reward_importance_weighted_accepted.item(),
            importance_weight * self.batch.reward[1, 0].item(),
            places=5,
        )
        self.assertEqual(eval_module.sum_reward_weighted_accepted_local.item(), 0.0)
        self.assertEqual(
            eval_module.sum_reward_importance_weighted_accepted_local.item(), 0.0
        )
        self.assertEqual(eval_module.get_avg_reward(), self.batch.reward[1, 0].item())

    def test_update_eval_model(self):
        policy_network_1 = LinearRegressionUCB(2)
        policy_network_1.avg_A += 0.3
        policy_network_2 = LinearRegressionUCB(2)
        policy_network_2.avg_A += 0.1
        eval_module = PolicyEvaluator(policy_network_1)
        self.assertTrue(
            _compare_state_dicts(
                eval_module.eval_model.state_dict(), policy_network_1.state_dict()
            )
        )

        eval_module.update_eval_model(policy_network_2)
        self.assertTrue(
            _compare_state_dicts(
                eval_module.eval_model.state_dict(), policy_network_2.state_dict()
            )
        )

        # change to the source model shouldn't affect the model in the eval module
        original_state_dict_2 = copy.deepcopy(policy_network_2.state_dict())
        policy_network_2.avg_A += 0.4
        self.assertTrue(
            _compare_state_dicts(
                eval_module.eval_model.state_dict(), original_state_dict_2
            )
        )

    def test_ingest_batch(self):
        model_actions = torch.tensor([[1], [1]], dtype=torch.long)
        _ = self.eval_module.ingest_batch(self.batch, model_actions)
        self.eval_module._aggregate_across_instances()
        # correct average reward
        self.assertEqual(
            self.eval_module.get_avg_reward(), self.batch.reward[1, 0].item()
        )

    def test_formatted_output(self):
        model_actions = torch.tensor([[1], [1]], dtype=torch.long)
        _ = self.eval_module.ingest_batch(self.batch, model_actions)
        self.eval_module._aggregate_across_instances()
        output = self.eval_module.get_formatted_result_string()
        self.assertIsInstance(output, str)

    def test_logger(self):
        logger = TensorBoardLogger("/tmp/tb")
        logger.log_metrics = MagicMock()
        self.eval_module.attach_logger(logger)
        self.eval_module.log_metrics(step=5)

        expected_metric_dict = {
            "[model]Offline_Eval_avg_reward": 0.0,
            "[model]Offline_Eval_sum_weight_accepted": 0.0,
            "[model]Offline_Eval_sum_weight_all_data": 0.0,
            "[model]Offline_Eval_num_eval_model_updates": 0,
            "[model]Offline_Eval_frac_accepted": 0.0,
            "[model]Offline_Eval_avg_reward_accepted": 0.0,
            "[model]Offline_Eval_avg_reward_rejected": 0.0,
            "[model]Offline_Eval_avg_size_accepted": 0.0,
            "[model]Offline_Eval_avg_size_rejected": 0.0,
            "[model]Offline_Eval_accepted_rejected_reward_ratio": 0.0,
            "[model]Offline_Eval_avg_reward_all_data": 0.0,
        }
        logger.log_metrics.assert_called_once_with(expected_metric_dict, step=5)
