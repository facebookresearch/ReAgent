#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from typing import List, Optional, Tuple

import torch
from reagent.core.parameters import EvaluationParameters, RLParameters
from reagent.core.torch_utils import masked_softmax
from reagent.core.types import DiscreteDqnInput, ExtraData, FeatureData
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.models.dqn import FullyConnectedDQN
from reagent.training.dqn_trainer_base import DQNTrainerBaseLightning
from reagent.training.parameters import DQNTrainerParameters
from reagent.workflow.types import RewardOptions


class MockDQNTrainer(DQNTrainerBaseLightning):
    """A minimal child class to test the methods in the DQNTrainerBase class."""

    def __init__(
        self,
        rl_parameters: RLParameters,
        metrics_to_score=None,
        actions: Optional[List[str]] = None,
        evaluation_parameters: Optional[EvaluationParameters] = None,
        double_q_learning: bool = True,
    ):
        super().__init__(
            rl_parameters,
            metrics_to_score=metrics_to_score,
            actions=actions,
            evaluation_parameters=evaluation_parameters,
        )
        self.double_q_learning = double_q_learning

    @torch.no_grad()
    def get_detached_model_outputs(
        self, state
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Gets the q values from the model and target networks"""
        q_values = self.q_network(state)
        q_values_target = self.q_network_target(state)
        return q_values, q_values_target


class TestDQNTrainerBaseLightning(unittest.TestCase):
    def setUp(self):
        self.params = DQNTrainerParameters(actions=["1", "2"])
        self.reward_options = RewardOptions()
        self.metrics_to_score = get_metrics_to_score(
            self.reward_options.metric_reward_values
        )
        self.state_dim = 10
        self.action_dim = 2
        self.batch_size = 3
        self.sizes = [20, 20]
        self.activations = ["relu", "relu"]
        self.q_network = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            sizes=self.sizes,
            activations=self.activations,
        )
        self.q_network_target = self.q_network.get_target_network()
        self.eval_parameters = EvaluationParameters(calc_cpe_in_training=True)
        self.num_output_nodes = (len(self.metrics_to_score) + 1) * len(
            self.params.actions
        )
        self.reward_network = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=self.num_output_nodes,
            sizes=self.sizes,
            activations=self.activations,
        )
        self.q_network_cpe = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=self.num_output_nodes,
            sizes=self.sizes,
            activations=self.activations,
        )
        self.q_network_cpe_target = self.q_network_cpe.get_target_network()

    def _construct_trainer(self, new_params=None, no_cpe=False):
        evaluation = self.eval_parameters
        params = self.params

        if new_params is not None:
            params = new_params
        if no_cpe:
            evaluation = EvaluationParameters(calc_cpe_in_training=False)

        return MockDQNTrainer(
            actions=params.actions,
            rl_parameters=params.rl,
            metrics_to_score=self.metrics_to_score,
            evaluation_parameters=evaluation,
        )

    def test_get_max_q_values_with_target(self):
        q_values = torch.tensor([[3.0, 4.0]])
        q_values_target = torch.tensor([[2.0, 1.0]])
        trainer = self._construct_trainer()

        # double q-learning (default)
        possible_actions_mask = torch.ones(q_values.shape)
        max_q_values_target, max_indicies = trainer.get_max_q_values_with_target(
            q_values, q_values_target, possible_actions_mask
        )
        self.assertEqual(max_indicies, torch.tensor([[1]]))
        self.assertEqual(max_q_values_target, torch.tensor([[1.0]]))

        # mask out max-value action
        possible_actions_mask = torch.tensor([[1, 0]])
        max_q_values_target, max_indicies = trainer.get_max_q_values_with_target(
            q_values, q_values_target, possible_actions_mask
        )
        self.assertEqual(max_indicies, torch.tensor([[0]]))
        self.assertEqual(max_q_values_target, torch.tensor([[2.0]]))

        # simple q-learning
        trainer.double_q_learning = False
        possible_actions_mask = torch.ones(q_values.shape)
        max_q_values_target, max_indicies = trainer.get_max_q_values_with_target(
            q_values, q_values_target, possible_actions_mask
        )
        self.assertEqual(max_indicies, torch.tensor([[0]]))
        self.assertEqual(max_q_values_target, torch.tensor([[2.0]]))

        # mask out max-value action
        possible_actions_mask = torch.tensor([[0, 1]])
        max_q_values_target, max_indicies = trainer.get_max_q_values_with_target(
            q_values, q_values_target, possible_actions_mask
        )
        self.assertEqual(max_indicies, torch.tensor([[1]]))
        self.assertEqual(max_q_values_target, torch.tensor([[1.0]]))

    def test_boost_rewards(self):
        rewards = torch.ones(3, 1)
        actions = torch.tensor([[0, 1], [1, 0], [0, 1]])
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(reward_boost={"1": 1.0, "2": 2.0})
        )
        trainer = self._construct_trainer(new_params=param_copy)
        boosted_reward = trainer.boost_rewards(rewards, actions)
        self.assertTrue(
            torch.equal(boosted_reward, torch.tensor([[3.0], [2.0], [3.0]]))
        )

    def test__initialize_cpe(self):
        reward_network = self.reward_network
        q_network_cpe = self.q_network_cpe
        q_network_cpe_target = self.q_network_cpe_target
        optimizer = self.params.optimizer
        # CPE
        trainer = self._construct_trainer()
        trainer._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, optimizer
        )
        self.assertTrue(torch.equal(trainer.reward_idx_offsets, torch.tensor([0])))
        self.assertIsNotNone(trainer.reward_network)
        self.assertIsNotNone(trainer.q_network_cpe)
        self.assertIsNotNone(trainer.q_network_cpe_target)
        self.assertIsNotNone(trainer.reward_network_optimizer)
        self.assertIsNotNone(trainer.q_network_cpe_optimizer)
        self.assertIsNotNone(trainer.evaluator)
        # no CPE
        trainer = self._construct_trainer(no_cpe=True)
        trainer._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, optimizer
        )
        self.assertIsNone(trainer.reward_network)

    def test__initialize_cpe_extra_metrics(self):
        reward_options = RewardOptions(
            metric_reward_values={"metric_a": 2, "metric_b": -2}
        )
        self.metrics_to_score = get_metrics_to_score(
            reward_options.metric_reward_values
        )
        num_output_nodes = (len(self.metrics_to_score) + 1) * len(self.params.actions)
        reward_network = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=num_output_nodes,
            sizes=self.sizes,
            activations=self.activations,
        )
        q_network_cpe = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=num_output_nodes,
            sizes=self.sizes,
            activations=self.activations,
        )
        q_network_cpe_target = q_network_cpe.get_target_network()
        reward_network = self.reward_network
        q_network_cpe = self.q_network_cpe
        q_network_cpe_target = self.q_network_cpe_target
        optimizer = self.params.optimizer
        # CPE
        trainer = self._construct_trainer()
        trainer._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, optimizer
        )
        self.assertTrue(
            torch.equal(trainer.reward_idx_offsets, torch.tensor([0, 2, 4]))
        )
        self.assertIsNotNone(trainer.reward_network)
        self.assertIsNotNone(trainer.q_network_cpe)
        self.assertIsNotNone(trainer.q_network_cpe_target)
        self.assertIsNotNone(trainer.reward_network_optimizer)
        self.assertIsNotNone(trainer.q_network_cpe_optimizer)
        self.assertIsNotNone(trainer.evaluator)
        # no CPE
        trainer = self._construct_trainer(no_cpe=True)
        trainer._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, optimizer
        )
        self.assertIsNone(trainer.reward_network)

    def test__configure_cpe_optimizers(self):
        reward_network = self.reward_network
        q_network_cpe = self.q_network_cpe
        q_network_cpe_target = self.q_network_cpe_target
        trainer = self._construct_trainer()
        trainer._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, self.params.optimizer
        )
        _, _, optimizers = trainer._configure_cpe_optimizers()
        # expecting a [reward_network_optimizer, q_network_cpe_optimizer] list
        self.assertEqual(len(optimizers), 2)

    def test__calculate_cpes(self):
        inp = DiscreteDqnInput(
            state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            next_state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            reward=torch.ones(self.batch_size, 1),
            time_diff=torch.ones(self.batch_size, 1) * 2,
            step=torch.ones(self.batch_size, 1) * 2,
            not_terminal=torch.ones(self.batch_size, 1),
            action=torch.tensor([[0, 1], [1, 0], [0, 1]]),
            next_action=torch.tensor([[1, 0], [0, 1], [1, 0]]),
            possible_actions_mask=torch.ones(self.batch_size, self.action_dim),
            possible_next_actions_mask=torch.ones(self.batch_size, self.action_dim),
            extras=ExtraData(),
        )

        mse_backward_type = type(
            torch.nn.functional.mse_loss(
                torch.tensor([1.0, 0.0], requires_grad=True), torch.zeros(1)
            ).grad_fn
        )
        trainer = self._construct_trainer()
        trainer._initialize_cpe(
            self.reward_network,
            self.q_network_cpe,
            self.q_network_cpe_target,
            self.params.optimizer,
        )
        trainer.reward_network = self.reward_network
        not_done_mask = inp.not_terminal.float()
        discount_tensor = torch.ones(inp.reward.shape)
        all_q_values = self.q_network(inp.state)
        all_action_scores = all_q_values.detach()
        all_next_action_scores = self.q_network(inp.next_state).detach()
        logged_action_idxs = torch.tensor([[1], [0], [1]])

        cpes_gen = trainer._calculate_cpes(
            inp,
            inp.state,
            inp.next_state,
            all_action_scores,
            all_next_action_scores,
            logged_action_idxs,
            discount_tensor,
            not_done_mask,
        )
        cpes = list(cpes_gen)

        ## expected reward_loss

        reward_target = inp.reward
        reward_estimate = trainer.reward_network(inp.state)
        # rewards at offset + logged action idx
        reward_estimate = reward_estimate.gather(1, torch.tensor([[1], [0], [1]]))
        mse_reward_loss = torch.nn.functional.mse_loss(reward_estimate, reward_target)

        ## expected metric_q_value_loss

        # assuming masked_softmax is tested elsewhere,
        # we can treat this as expected ground truth value
        model_propensities_next_states = masked_softmax(
            all_next_action_scores,
            inp.possible_next_actions_mask
            if trainer.maxq_learning
            else inp.next_action,
            trainer.rl_temperature,
        )
        metric_q_values = trainer.q_network_cpe(inp.state).gather(
            1, torch.tensor([[1], [0], [1]])
        )
        metrics_target_q_values = trainer.q_network_cpe_target(inp.next_state)
        per_metric_next_q_values = torch.sum(
            metrics_target_q_values * model_propensities_next_states,
            1,
            keepdim=True,
        )
        per_metric_next_q_values *= not_done_mask
        metrics_target_q_values = (
            reward_target + discount_tensor * per_metric_next_q_values
        )
        metric_q_value_loss = trainer.q_network_loss(
            metric_q_values, metrics_target_q_values
        )
        self.assertEqual(len(cpes), 2)
        self.assertEqual(type(cpes[0].grad_fn), mse_backward_type)
        self.assertEqual(type(cpes[1].grad_fn), mse_backward_type)
        self.assertEqual(cpes[0], mse_reward_loss)
        self.assertEqual(cpes[1], metric_q_value_loss)

    def test__calculate_cpes_extra_metrics(self):
        reward_options = RewardOptions(
            metric_reward_values={"metric_a": 2, "metric_b": -2}
        )
        self.metrics_to_score = get_metrics_to_score(
            reward_options.metric_reward_values
        )
        num_output_nodes = (len(self.metrics_to_score) + 1) * len(self.params.actions)
        # re-initialize networks with larger output layers
        reward_network = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=num_output_nodes,
            sizes=self.sizes,
            activations=self.activations,
        )
        q_network_cpe = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=num_output_nodes,
            sizes=self.sizes,
            activations=self.activations,
        )
        q_network_cpe_target = q_network_cpe.get_target_network()
        # mock data for two extra metrcis: a and b
        extra_metrics = torch.concat(
            (2 * torch.ones(self.batch_size, 1), -2 * torch.ones(self.batch_size, 1)),
            dim=1,
        )
        # initialize batch with extra metrics data
        inp = DiscreteDqnInput(
            state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            next_state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            reward=torch.ones(self.batch_size, 1),
            time_diff=torch.ones(self.batch_size, 1) * 2,
            step=torch.ones(self.batch_size, 1) * 2,
            not_terminal=torch.ones(self.batch_size, 1),
            action=torch.tensor([[0, 1], [1, 0], [0, 1]]),
            next_action=torch.tensor([[1, 0], [0, 1], [1, 0]]),
            possible_actions_mask=torch.ones(self.batch_size, self.action_dim),
            possible_next_actions_mask=torch.ones(self.batch_size, self.action_dim),
            extras=ExtraData(metrics=extra_metrics),
        )

        mse_backward_type = type(
            torch.nn.functional.mse_loss(
                torch.tensor([1.0, 0.0], requires_grad=True), torch.zeros(1)
            ).grad_fn
        )
        trainer = self._construct_trainer()
        trainer._initialize_cpe(
            reward_network,
            q_network_cpe,
            q_network_cpe_target,
            self.params.optimizer,
        )
        trainer.reward_network = reward_network
        not_done_mask = inp.not_terminal.float()
        discount_tensor = torch.ones(inp.reward.shape)
        all_q_values = self.q_network(inp.state)
        all_action_scores = all_q_values.detach()
        all_next_action_scores = self.q_network(inp.next_state).detach()
        logged_action_idxs = torch.tensor([[1], [0], [1]])

        cpes_gen = trainer._calculate_cpes(
            inp,
            inp.state,
            inp.next_state,
            all_action_scores,
            all_next_action_scores,
            logged_action_idxs,
            discount_tensor,
            not_done_mask,
        )
        cpes = list(cpes_gen)
        # offset + logged action idx tensor
        offset_tensor = torch.tensor([[1, 3, 5], [0, 2, 4], [1, 3, 5]])
        ## expected reward_loss

        reward_target = torch.cat((inp.reward, inp.extras.metrics), dim=1)
        reward_estimate = trainer.reward_network(inp.state)
        reward_estimate = reward_estimate.gather(1, offset_tensor)
        mse_reward_loss = torch.nn.functional.mse_loss(reward_estimate, reward_target)

        ## expected metric_q_value_loss

        model_propensities_next_states = masked_softmax(
            all_next_action_scores,
            inp.possible_next_actions_mask
            if trainer.maxq_learning
            else inp.next_action,
            trainer.rl_temperature,
        )
        # q_values at offset + logged action idx
        metric_q_values = trainer.q_network_cpe(inp.state).gather(1, offset_tensor)
        metrics_target_q_values = torch.chunk(
            trainer.q_network_cpe_target(inp.next_state),
            3,
            dim=1,
        )
        target_metric_q_values = []
        for i, per_metric_target_q_values in enumerate(metrics_target_q_values):
            per_metric_next_q_values = torch.sum(
                per_metric_target_q_values * model_propensities_next_states,
                1,
                keepdim=True,
            )
            per_metric_next_q_values = per_metric_next_q_values * not_done_mask
            per_metric_target_q_values = reward_target[:, i : i + 1] + (
                discount_tensor * per_metric_next_q_values
            )
            target_metric_q_values.append(per_metric_target_q_values)

        target_metric_q_values = torch.cat(target_metric_q_values, dim=1)
        metric_q_value_loss = trainer.q_network_loss(
            metric_q_values, target_metric_q_values
        )

        self.assertEqual(len(cpes), 2)
        self.assertEqual(type(cpes[0].grad_fn), mse_backward_type)
        self.assertEqual(type(cpes[1].grad_fn), mse_backward_type)
        self.assertEqual(cpes[0], mse_reward_loss)
        self.assertEqual(cpes[1], metric_q_value_loss)

    def test_gather_eval_data(self):
        batch_num = 2
        batches = []
        # generate several data batches
        for _ in range(batch_num):
            inp = DiscreteDqnInput(
                state=FeatureData(
                    float_features=torch.rand(self.batch_size, self.state_dim)
                ),
                next_state=FeatureData(
                    float_features=torch.rand(self.batch_size, self.state_dim)
                ),
                reward=torch.ones(self.batch_size, 1),
                time_diff=torch.ones(self.batch_size, 1) * 2,
                step=torch.ones(self.batch_size, 1) * 2,
                not_terminal=torch.ones(self.batch_size, 1),
                action=torch.tensor([[0, 1], [1, 0], [0, 1]]),
                next_action=torch.tensor([[1, 0], [0, 1], [1, 0]]),
                possible_actions_mask=torch.ones(self.batch_size, self.action_dim),
                possible_next_actions_mask=torch.ones(self.batch_size, self.action_dim),
                extras=ExtraData(),
            )
            batches.append(inp)
        trainer = self._construct_trainer()
        trainer.q_network = self.q_network
        trainer.q_network_target = self.q_network_target
        trainer.q_network_cpe = self.q_network_cpe
        trainer.reward_network = self.reward_network
        # generate evaluation datapages for each batch
        data_pages = [
            trainer.validation_step(batch=inp, batch_idx=idx + 1)
            for idx, inp in enumerate(batches)
        ]
        # aggregate datapages
        eval_data = trainer.gather_eval_data(data_pages)
        self.assertEqual(
            eval_data.model_rewards.shape[0],
            batch_num * data_pages[0].model_rewards.shape[0],
        )
        self.assertEqual(
            eval_data.logged_rewards.shape[0],
            batch_num * data_pages[0].logged_rewards.shape[0],
        )
        self.assertEqual(
            eval_data.action_mask.shape[0],
            batch_num * data_pages[0].action_mask.shape[0],
        )
        self.assertEqual(
            eval_data.model_propensities.shape[0],
            batch_num * data_pages[0].model_propensities.shape[0],
        )
        self.assertEqual(
            eval_data.model_values.shape[0],
            batch_num * data_pages[0].model_values.shape[0],
        )
        self.assertEqual(
            eval_data.possible_actions_mask.shape[0],
            batch_num * data_pages[0].possible_actions_mask.shape[0],
        )
        self.assertEqual(
            eval_data.optimal_q_values.shape[0],
            batch_num * data_pages[0].optimal_q_values.shape[0],
        )
        self.assertEqual(
            eval_data.eval_action_idxs.shape[0],
            batch_num * data_pages[0].eval_action_idxs.shape[0],
        )

    def test_validation_step(self):
        inp = DiscreteDqnInput(
            state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            next_state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            reward=torch.ones(self.batch_size, 1),
            time_diff=torch.ones(self.batch_size, 1) * 2,
            step=torch.ones(self.batch_size, 1) * 2,
            not_terminal=torch.ones(self.batch_size, 1),
            action=torch.tensor([[0, 1], [1, 0], [0, 1]]),
            next_action=torch.tensor([[1, 0], [0, 1], [1, 0]]),
            possible_actions_mask=torch.ones(self.batch_size, self.action_dim),
            possible_next_actions_mask=torch.ones(self.batch_size, self.action_dim),
            extras=ExtraData(),
        )
        trainer = self._construct_trainer()
        trainer.q_network = self.q_network
        trainer.q_network_target = self.q_network_target
        trainer.q_network_cpe = self.q_network_cpe
        trainer.reward_network = self.reward_network
        data_page = trainer.validation_step(batch=inp, batch_idx=1)
        self.assertTrue(isinstance(data_page, EvaluationDataPage))
