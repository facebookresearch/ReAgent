#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.core.parameters import EvaluationParameters, RLParameters
from reagent.core.types import DiscreteDqnInput, ExtraData, FeatureData
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.models.dqn import FullyConnectedDQN
from reagent.training.dqn_trainer import DQNTrainer
from reagent.training.parameters import DQNTrainerParameters
from reagent.workflow.types import RewardOptions


class TestDQN(unittest.TestCase):
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
        self.x = FeatureData(float_features=torch.rand(5, self.state_dim))
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
        reward_network = self.reward_network
        q_network_cpe = self.q_network_cpe
        q_network_cpe_target = self.q_network_cpe_target
        evaluation = self.eval_parameters
        params = self.params

        if new_params is not None:
            params = new_params
        if no_cpe:
            reward_network = q_network_cpe = q_network_cpe_target = None
            evaluation = EvaluationParameters(calc_cpe_in_training=False)

        return DQNTrainer(
            q_network=self.q_network,
            q_network_target=self.q_network_target,
            reward_network=reward_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=self.metrics_to_score,
            evaluation=evaluation,
            **params.asdict(),
        )

    def test_init(self):
        trainer = self._construct_trainer()
        self.assertTrue((torch.isclose(trainer.reward_boosts, torch.zeros(2))).all())
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(reward_boost={"1": 1, "2": 2})
        )
        reward_boost_trainer = self._construct_trainer(new_params=param_copy)
        self.assertTrue(
            (
                torch.isclose(
                    reward_boost_trainer.reward_boosts, torch.tensor([1.0, 2.0])
                )
            ).all()
        )

    def test_train_step_gen(self):
        # mock training batch
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
        add_backward_type = type(
            (
                torch.tensor([1.0], requires_grad=True)
                + torch.tensor([1.0], requires_grad=True)
            ).grad_fn
        )

        # vanilla DQN with CPE
        trainer = self._construct_trainer()
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        # four outputs of the train_step_gen method call:
        # td_loss, two CPE losses (reward_loss and metric_q_value_loss),
        # and soft_update_result
        self.assertEqual(len(losses), 4)
        self.assertEqual(type(losses[0].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[1].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[2].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[3].grad_fn), add_backward_type)

        # no CPE
        trainer = self._construct_trainer(no_cpe=True)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        # two outputs of the train_step_gen method with no CPE
        self.assertEqual(len(losses), 2)

        # seq_num
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(use_seq_num_diff_as_time_diff=True)
        )
        trainer = self._construct_trainer(new_params=param_copy)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)

        # multi_steps
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(multi_steps=2)
        )
        trainer = self._construct_trainer(new_params=param_copy)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)

        # non_max_q
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(maxq_learning=False)
        )
        trainer = self._construct_trainer(new_params=param_copy)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)

    def test_configure_optimizers(self):
        trainer = self._construct_trainer()
        optimizers = trainer.configure_optimizers()
        # expecting a list of [
        #   q_network optimizer,
        #   reward_network optimizer,
        #   q_network_cpe optimizer,
        #   soft_update optimizer]
        self.assertEqual(len(optimizers), 4)
        train_step_yield_order = [
            trainer.q_network,
            trainer.reward_network,
            trainer.q_network_cpe,
            trainer.q_network,
        ]
        for i in range(len(train_step_yield_order)):
            opt_param = optimizers[i]["optimizer"].param_groups[0]["params"][0]
            loss_param = list(train_step_yield_order[i].parameters())[0]
            self.assertTrue(torch.all(torch.isclose(opt_param, loss_param)))

        trainer = self._construct_trainer(no_cpe=True)
        optimizers = trainer.configure_optimizers()
        # expecting a [q_network optimizer, soft_update optimizer] list
        self.assertEqual(len(optimizers), 2)

    def test_get_detached_model_outputs(self):
        trainer = self._construct_trainer()
        q_out, q_target = trainer.get_detached_model_outputs(self.x)
        self.assertEqual(q_out.shape[0], q_target.shape[0], self.batch_size)
        self.assertEqual(q_out.shape[1], q_target.shape[1], self.action_dim)

    def test_compute_discount_tensor(self):
        time_diff = 4
        steps = 3
        inp = DiscreteDqnInput(
            state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            next_state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            reward=torch.ones(self.batch_size, 1),
            time_diff=torch.ones(self.batch_size, 1) * time_diff,
            step=torch.ones(self.batch_size, 1) * steps,
            not_terminal=torch.ones(self.batch_size, 1),
            action=torch.tensor([[0, 1], [1, 0], [0, 1]]),
            next_action=torch.tensor([[1, 0], [0, 1], [1, 0]]),
            possible_actions_mask=torch.ones(self.batch_size, self.action_dim),
            possible_next_actions_mask=torch.ones(self.batch_size, self.action_dim),
            extras=ExtraData(),
        )

        # vanilla
        trainer = self._construct_trainer()
        discount_tensor = trainer.compute_discount_tensor(
            batch=inp, boosted_rewards=inp.reward
        )
        self.assertEqual(discount_tensor.shape[0], self.batch_size)
        self.assertEqual(discount_tensor.shape[1], 1)
        self.assertTrue(
            torch.isclose(discount_tensor, torch.tensor(trainer.gamma)).all()
        )

        # seq_num
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(use_seq_num_diff_as_time_diff=True)
        )
        trainer = self._construct_trainer(new_params=param_copy)
        discount_tensor = trainer.compute_discount_tensor(
            batch=inp, boosted_rewards=inp.reward
        )
        self.assertEqual(discount_tensor.shape[0], self.batch_size)
        self.assertEqual(discount_tensor.shape[1], 1)
        self.assertTrue(
            torch.isclose(discount_tensor, torch.tensor(trainer.gamma**time_diff)).all()
        )

        # multi_steps
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(multi_steps=steps)
        )
        trainer = self._construct_trainer(new_params=param_copy)
        discount_tensor = trainer.compute_discount_tensor(
            batch=inp, boosted_rewards=inp.reward
        )
        self.assertEqual(discount_tensor.shape[0], self.batch_size)
        self.assertEqual(discount_tensor.shape[1], 1)
        self.assertTrue(
            torch.isclose(discount_tensor, torch.tensor(trainer.gamma**steps)).all()
        )

    def test_compute_td_loss(self):
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

        # vanilla mse loss
        trainer = self._construct_trainer()
        discount_tensor = trainer.compute_discount_tensor(
            batch=inp, boosted_rewards=inp.reward
        )
        mse_backward_type = type(
            torch.nn.functional.mse_loss(
                torch.tensor([1.0, 0.0], requires_grad=True), torch.zeros(1)
            ).grad_fn
        )
        loss = trainer.compute_td_loss(
            batch=inp, boosted_rewards=inp.reward, discount_tensor=discount_tensor
        )
        self.assertEqual(type(loss.grad_fn), mse_backward_type)

        # huber loss
        param_copy = DQNTrainerParameters(
            actions=["1", "2"], rl=RLParameters(q_network_loss="huber")
        )
        trainer = self._construct_trainer(new_params=param_copy)
        discount_tensor = trainer.compute_discount_tensor(
            batch=inp, boosted_rewards=inp.reward
        )
        smooth_l1_backward_type = type(
            torch.nn.functional.smooth_l1_loss(
                torch.tensor([1.0], requires_grad=True), torch.zeros(1)
            ).grad_fn
        )
        loss = trainer.compute_td_loss(
            batch=inp, boosted_rewards=inp.reward, discount_tensor=discount_tensor
        )
        self.assertEqual(type(loss.grad_fn), smooth_l1_backward_type)

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
        data_page = trainer.validation_step(batch=inp, batch_idx=1)
        self.assertTrue(isinstance(data_page, EvaluationDataPage))

    def test__dense_to_action_dict(self):
        trainer = self._construct_trainer()
        dense = torch.rand(trainer.num_actions)
        retval = trainer._dense_to_action_dict(dense)
        self.assertEqual(len(retval), trainer.num_actions)
        for i, a in enumerate(self.params.actions):
            self.assertTrue(a in retval)
            self.assertTrue(torch.isclose(retval[a], dense[i]))
