#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.core.parameters import EvaluationParameters, RLParameters
from reagent.core.types import DiscreteDqnInput, ExtraData, FeatureData
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.models.dqn import FullyConnectedDQN
from reagent.training.parameters import QRDQNTrainerParameters
from reagent.training.qrdqn_trainer import QRDQNTrainer
from reagent.workflow.types import RewardOptions


class TestQRDQN(unittest.TestCase):
    def setUp(self):
        # preparing various components for qr-dqn trainer initialization
        self.params = QRDQNTrainerParameters(actions=["1", "2"], num_atoms=11)
        self.reward_options = RewardOptions()
        self.metrics_to_score = get_metrics_to_score(
            self.reward_options.metric_reward_values
        )
        self.state_dim = 10
        self.action_dim = 2
        self.sizes = [20, 20]
        self.num_atoms = 11
        self.activations = ["relu", "relu"]
        self.dropout_ratio = 0
        self.q_network = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            sizes=self.sizes,
            num_atoms=self.num_atoms,
            activations=self.activations,
            dropout_ratio=self.dropout_ratio,
        )
        self.q_network_target = self.q_network.get_target_network()
        self.x = FeatureData(float_features=torch.rand(5, 10))
        self.eval_parameters = EvaluationParameters(calc_cpe_in_training=True)
        self.num_output_nodes = (len(self.metrics_to_score) + 1) * len(
            # pyre-fixme[16]: `QRDQNTrainerParameters` has no attribute `actions`.
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

        return QRDQNTrainer(
            q_network=self.q_network,
            q_network_target=self.q_network_target,
            reward_network=reward_network,
            q_network_cpe=q_network_cpe,
            q_network_cpe_target=q_network_cpe_target,
            metrics_to_score=self.metrics_to_score,
            evaluation=evaluation,
            # pyre-fixme[16]: `QRDQNTrainerParameters` has no attribute `asdict`.
            **params.asdict()
        )

    def test_init(self):
        trainer = self._construct_trainer()
        quantiles = (0.5 + torch.arange(self.num_atoms).float()) / float(self.num_atoms)
        self.assertTrue((torch.isclose(trainer.quantiles, quantiles)).all())
        self.assertTrue((torch.isclose(trainer.reward_boosts, torch.zeros(2))).all())
        param_copy = QRDQNTrainerParameters(
            actions=["1", "2"],
            num_atoms=11,
            rl=RLParameters(reward_boost={"1": 1, "2": 2}),
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
        inp = DiscreteDqnInput(
            state=FeatureData(float_features=torch.rand(3, 10)),
            next_state=FeatureData(float_features=torch.rand(3, 10)),
            reward=torch.ones(3, 1),
            time_diff=torch.ones(3, 1) * 2,
            step=torch.ones(3, 1) * 2,
            not_terminal=torch.ones(3, 1),  # todo: check terminal behavior
            action=torch.tensor([[0, 1], [1, 0], [0, 1]]),
            next_action=torch.tensor([[1, 0], [0, 1], [1, 0]]),
            possible_actions_mask=torch.ones(3, 2),
            possible_next_actions_mask=torch.ones(3, 2),
            extras=ExtraData(),
        )
        mse_backward_type = type(
            torch.nn.functional.mse_loss(
                torch.tensor([1.0], requires_grad=True), torch.zeros(1)
            ).grad_fn
        )
        add_backward_type = type(
            (
                torch.tensor([1.0], requires_grad=True)
                + torch.tensor([1.0], requires_grad=True)
            ).grad_fn
        )
        mean_backward_type = type(
            torch.tensor([1.0, 2.0], requires_grad=True).mean().grad_fn
        )

        # vanilla
        trainer = self._construct_trainer()
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)
        self.assertEqual(type(losses[0].grad_fn), mean_backward_type)
        self.assertEqual(type(losses[1].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[2].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[3].grad_fn), add_backward_type)

        # no CPE
        trainer = self._construct_trainer(no_cpe=True)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 2)

        # seq_num
        param_copy = QRDQNTrainerParameters(
            actions=["1", "2"],
            num_atoms=11,
            rl=RLParameters(use_seq_num_diff_as_time_diff=True),
        )
        trainer = self._construct_trainer(new_params=param_copy)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)

        # multi_steps
        param_copy = QRDQNTrainerParameters(
            actions=["1", "2"], num_atoms=11, rl=RLParameters(multi_steps=2)
        )
        trainer = self._construct_trainer(new_params=param_copy)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)

        # non_max_q
        param_copy = QRDQNTrainerParameters(
            actions=["1", "2"], num_atoms=11, rl=RLParameters(maxq_learning=False)
        )
        trainer = self._construct_trainer(new_params=param_copy)
        loss_gen = trainer.train_step_gen(inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)

    def test_configure_optimizers(self):
        trainer = self._construct_trainer()
        optimizers = trainer.configure_optimizers()
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
        self.assertEqual(len(optimizers), 2)

    def test_get_detached_model_outputs(self):
        trainer = self._construct_trainer()
        q_out, q_target = trainer.get_detached_model_outputs(self.x)
        self.assertEqual(q_out.shape[0], q_target.shape[0], 3)
        self.assertEqual(q_out.shape[1], q_target.shape[1], 2)
