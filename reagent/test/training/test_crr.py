#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.core.parameters import EvaluationParameters, RLParameters
from reagent.core.types import FeatureData, DiscreteDqnInput, ExtraData
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.models.actor import FullyConnectedActor
from reagent.models.dqn import FullyConnectedDQN
from reagent.training.discrete_crr_trainer import DiscreteCRRTrainer
from reagent.training.parameters import CRRTrainerParameters
from reagent.workflow.types import RewardOptions


class TestCRR(unittest.TestCase):
    def setUp(self):
        # preparing various components for qr-dqn trainer initialization
        self.batch_size = 3
        self.state_dim = 10
        self.action_dim = 2
        self.num_layers = 2
        self.sizes = [20 for _ in range(self.num_layers)]
        self.num_atoms = 11
        self.activations = ["relu" for _ in range(self.num_layers)]
        self.dropout_ratio = 0
        self.exploration_variance = 1e-10

        self.actions = [str(i) for i in range(self.action_dim)]
        self.params = CRRTrainerParameters(actions=self.actions)
        self.reward_options = RewardOptions()
        self.metrics_to_score = get_metrics_to_score(
            self.reward_options.metric_reward_values
        )

        self.actor_network = FullyConnectedActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            sizes=self.sizes,
            activations=self.activations,
            exploration_variance=self.exploration_variance,
        )
        self.actor_network_target = self.actor_network.get_target_network()

        self.q1_network = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            sizes=self.sizes,
            activations=self.activations,
            dropout_ratio=self.dropout_ratio,
        )
        self.q1_network_target = self.q1_network.get_target_network()

        self.q2_network = FullyConnectedDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            sizes=self.sizes,
            activations=self.activations,
            dropout_ratio=self.dropout_ratio,
        )
        self.q2_network_target = self.q2_network.get_target_network()

        self.num_output_nodes = (len(self.metrics_to_score) + 1) * len(
            self.params.actions
        )
        self.eval_parameters = EvaluationParameters(calc_cpe_in_training=True)
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
        self.inp = DiscreteDqnInput(
            state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            next_state=FeatureData(
                float_features=torch.rand(self.batch_size, self.state_dim)
            ),
            reward=torch.ones(self.batch_size, 1),
            time_diff=torch.ones(self.batch_size, 1) * 2,
            step=torch.ones(self.batch_size, 1) * 2,
            not_terminal=torch.ones(
                self.batch_size, 1
            ),  # todo: check terminal behavior
            action=torch.tensor([[0, 1], [1, 0], [0, 1]]),
            next_action=torch.tensor([[1, 0], [0, 1], [1, 0]]),
            possible_actions_mask=torch.ones(self.batch_size, self.action_dim),
            possible_next_actions_mask=torch.ones(self.batch_size, self.action_dim),
            extras=ExtraData(action_probability=torch.ones(self.batch_size, 1)),
        )

    @staticmethod
    def dummy_log(*args, **kwargs):
        # replaces calls to self.log() which otherwise require the pytorch lighting trainer to be intialized
        return None

    def _construct_trainer(self, new_params=None, no_cpe=False, no_q2=False):
        trainer = DiscreteCRRTrainer(
            actor_network=self.actor_network,
            actor_network_target=self.actor_network_target,
            q1_network=self.q1_network,
            q1_network_target=self.q1_network_target,
            q2_network=(None if no_q2 else self.q2_network),
            q2_network_target=(None if no_q2 else self.q2_network_target),
            reward_network=(None if no_cpe else self.reward_network),
            q_network_cpe=(None if no_cpe else self.q_network_cpe),
            q_network_cpe_target=(None if no_cpe else self.q_network_cpe_target),
            metrics_to_score=self.metrics_to_score,
            evaluation=EvaluationParameters(
                calc_cpe_in_training=(False if no_cpe else True)
            ),
            # pyre-fixme[16]: `QRDQNTrainerParameters` has no attribute `asdict`.
            **(new_params if new_params is not None else self.params).asdict()
        )
        trainer.log = self.dummy_log
        return trainer

    def test_init(self):
        trainer = self._construct_trainer()
        self.assertTrue((torch.isclose(trainer.reward_boosts, torch.zeros(2))).all())
        param_copy = CRRTrainerParameters(
            actions=self.actions,
            rl=RLParameters(reward_boost={i: int(i) + 1 for i in self.actions}),
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
        # vanilla
        trainer = self._construct_trainer()
        loss_gen = trainer.train_step_gen(self.inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 6)
        self.assertEqual(type(losses[0].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[1].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[2].grad_fn), add_backward_type)
        self.assertEqual(type(losses[3].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[4].grad_fn), mse_backward_type)
        self.assertEqual(type(losses[5].grad_fn), add_backward_type)

        # no CPE
        trainer = self._construct_trainer(no_cpe=True)
        loss_gen = trainer.train_step_gen(self.inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 4)

        # no q2 net
        trainer = self._construct_trainer(no_q2=True)
        loss_gen = trainer.train_step_gen(self.inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 5)

        # use_target_actor
        params_copy = CRRTrainerParameters(actions=self.actions, use_target_actor=True)
        trainer = self._construct_trainer(new_params=params_copy)
        loss_gen = trainer.train_step_gen(self.inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 6)

        # delayed policy update
        params_copy = CRRTrainerParameters(
            actions=self.actions, delayed_policy_update=2
        )
        trainer = self._construct_trainer(new_params=params_copy)
        loss_gen = trainer.train_step_gen(self.inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 6)
        self.assertEqual(losses[2], None)

        # entropy
        params_copy = CRRTrainerParameters(actions=self.actions, entropy_coeff=1.0)
        trainer = self._construct_trainer(new_params=params_copy)
        loss_gen = trainer.train_step_gen(self.inp, batch_idx=1)
        losses = list(loss_gen)
        self.assertEqual(len(losses), 6)

    def test_q_network_property(self):
        trainer = self._construct_trainer()
        self.assertEqual(trainer.q_network, trainer.q1_network)

    def test_configure_optimizers(self):
        trainer = self._construct_trainer()
        optimizers = trainer.configure_optimizers()
        self.assertEqual(len(optimizers), 6)
        train_step_yield_order = [
            trainer.q1_network,
            trainer.q2_network,
            trainer.actor_network,
            trainer.reward_network,
            trainer.q_network_cpe,
            trainer.q1_network,
        ]
        for i in range(len(train_step_yield_order)):
            opt_param = optimizers[i]["optimizer"].param_groups[0]["params"][0]
            loss_param = list(train_step_yield_order[i].parameters())[0]
            self.assertTrue(torch.all(torch.isclose(opt_param, loss_param)))
        trainer = self._construct_trainer(no_cpe=True)
        optimizers = trainer.configure_optimizers()
        self.assertEqual(len(optimizers), 4)
        trainer = self._construct_trainer(no_q2=True)
        optimizers = trainer.configure_optimizers()
        self.assertEqual(len(optimizers), 5)

    def test_get_detached_model_outputs(self):
        trainer = self._construct_trainer()
        action_scores, _ = trainer.get_detached_model_outputs(
            FeatureData(float_features=torch.rand(self.batch_size, self.state_dim))
        )
        self.assertEqual(action_scores.shape[0], self.batch_size)
        self.assertEqual(action_scores.shape[1], self.action_dim)

    def test_validation_step(self):
        trainer = self._construct_trainer()
        edp = trainer.validation_step(self.inp, batch_idx=1)
        out = trainer.actor_network(self.inp.state)
        # Note: in current code EDP assumes policy induced by q-net instead of actor
        self.assertTrue(torch.all(torch.isclose(edp.optimal_q_values, out.action)))
