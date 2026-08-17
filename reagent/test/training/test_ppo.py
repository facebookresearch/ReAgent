#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-strict

import unittest
from collections import defaultdict
from typing import Optional
from unittest import mock

import torch
from reagent.core.types import FeatureData, PolicyGradientInput
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.models.dueling_q_network import DuelingQNetwork
from reagent.models.fully_connected_network import FloatFeatureFullyConnected
from reagent.training.parameters import PPOTrainerParameters
from reagent.training.ppo_trainer import PPOTrainer
from reagent.training.utils import discounted_returns
from reagent.workflow.types import RewardOptions


class TestPPO(unittest.TestCase):
    def _params(self, **kwargs: object) -> PPOTrainerParameters:
        return PPOTrainerParameters(**kwargs)

    def setUp(self) -> None:
        # preparing various components for qr-dqn trainer initialization
        self.batch_size = 3
        self.state_dim = 10
        self.action_dim = 2
        self.num_layers = 2
        self.sizes = [20 for _ in range(self.num_layers)]
        self.activations = ["relu" for _ in range(self.num_layers)]
        self.use_layer_norm = False
        self.softmax_temperature = 1

        self.actions = [str(i) for i in range(self.action_dim)]
        self.params = self._params(actions=self.actions, normalize=False)
        self.reward_options = RewardOptions()
        self.metrics_to_score = get_metrics_to_score(
            self.reward_options.metric_reward_values
        )

        self.policy_network = DuelingQNetwork.make_fully_connected(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            layers=self.sizes,
            activations=self.activations,
        )
        self.sampler = SoftmaxActionSampler(temperature=self.softmax_temperature)
        self.policy = Policy(scorer=self.policy_network, sampler=self.sampler)

        self.value_network = FloatFeatureFullyConnected(
            state_dim=self.state_dim,
            output_dim=1,
            sizes=self.sizes,
            activations=self.activations,
            use_layer_norm=self.use_layer_norm,
        )

    def _construct_trainer(
        self,
        new_params: Optional[PPOTrainerParameters] = None,
        use_value_net: bool = True,
    ) -> PPOTrainer:
        value_network = self.value_network if use_value_net else None
        params = new_params if new_params else self.params

        trainer = PPOTrainer(
            policy=self.policy, value_net=value_network, **params.asdict()
        )
        trainer.optimizers = mock.Mock(return_value=[0, 0])
        return trainer

    def test_init(self) -> None:
        trainer = self._construct_trainer()

        self.assertEqual(
            type(trainer.value_loss_fn), type(torch.nn.MSELoss(reduction="sum"))
        )
        self.assertEqual(trainer.value_loss_fn.reduction, "sum")

        new_params = self._params(ppo_epsilon=-1)
        with self.assertRaises(AssertionError):
            self._construct_trainer(new_params)

        new_params = self._params(ppo_epsilon=2)
        with self.assertRaises(AssertionError):
            self._construct_trainer(new_params)

        params = self._params(actions=["1", "2"], normalize=True)
        with self.assertRaises(AssertionError):
            self._construct_trainer(new_params=params)

    def test__trajectory_to_losses(self) -> None:
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        # Normalize + offset clamp min
        params = self._params(
            actions=["1", "2"],
            normalize=True,
            offset_clamp_min=True,
        )
        trainer = self._construct_trainer(new_params=params, use_value_net=False)
        losses = trainer._trajectory_to_losses(inp)
        self.assertEqual(len(losses), 1)
        self.assertTrue("ppo_loss" in losses)

        trainer = self._construct_trainer()
        losses = trainer._trajectory_to_losses(inp)
        self.assertEqual(len(losses), 2)
        self.assertTrue("ppo_loss" in losses and "value_net_loss" in losses)
        # entropy weight should always lower ppo_loss
        trainer.entropy_weight = 1.0
        entropy_losses = trainer._trajectory_to_losses(inp)
        self.assertTrue(entropy_losses["ppo_loss"] < losses["ppo_loss"])

    def test_td_error_advantage_requires_value_net(self) -> None:
        params = self._params(
            actions=self.actions, normalize=False, td_error_advantage=True
        )
        with self.assertRaises(AssertionError):
            self._construct_trainer(new_params=params, use_value_net=False)

    def test_td_error_advantage(self) -> None:
        params = self._params(
            actions=self.actions, normalize=False, td_error_advantage=True
        )
        trainer = self._construct_trainer(new_params=params)
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        # Deterministic V(s_t) so the advantage is fully determined. value_net is
        # a registered submodule, so patch its forward rather than replacing it.
        values = torch.tensor([1.0, 2.0, 3.0])
        value_net = trainer.value_net
        assert value_net is not None
        value_net.forward = mock.Mock(return_value=values)
        # Force the importance ratio to 1 (target propensity == logged log_prob),
        # so ppo_loss reduces to -sum(advantage).
        trainer.sampler.log_prob = mock.Mock(return_value=inp.log_prob.detach())

        losses = trainer._trajectory_to_losses(inp)
        self.assertIn("ppo_loss", losses)
        self.assertIn("value_net_loss", losses)

        # A_t = r_t + gamma * V(s_{t+1}) - V(s_t), with V(s_T) = 0.
        rewards = torch.clamp(inp.reward.detach(), max=trainer.reward_clip)
        next_values = torch.cat([values[1:], values.new_zeros(1)])
        expected_advantage = rewards + trainer.gamma * next_values - values
        self.assertTrue(torch.allclose(losses["ppo_loss"], -expected_advantage.sum()))

    def test_td_error_advantage_offset_clamp_min(self) -> None:
        params = self._params(
            actions=self.actions,
            normalize=False,
            td_error_advantage=True,
            offset_clamp_min=True,
        )
        trainer = self._construct_trainer(new_params=params)
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        # Large V(s) forces every TD advantage negative, so offset_clamp_min
        # clamps them all to 0 (ppo_loss == 0).
        values = torch.tensor([100.0, 100.0, 100.0])
        value_net = trainer.value_net
        assert value_net is not None
        value_net.forward = mock.Mock(return_value=values)
        trainer.sampler.log_prob = mock.Mock(return_value=inp.log_prob.detach())

        losses = trainer._trajectory_to_losses(inp)

        rewards = torch.clamp(inp.reward.detach(), max=trainer.reward_clip)
        next_values = torch.cat([values[1:], values.new_zeros(1)])
        not_terminal = torch.tensor([1.0, 1.0, 0.0])
        td_target = rewards + trainer.gamma * not_terminal * next_values
        expected_advantage = (td_target - values).clamp(min=0)
        self.assertTrue(torch.allclose(losses["ppo_loss"], -expected_advantage.sum()))
        self.assertTrue(torch.allclose(losses["ppo_loss"], torch.zeros(())))

    def test_td_error_advantage_truncated_trajectory(self) -> None:
        params = self._params(
            actions=self.actions, normalize=False, td_error_advantage=True
        )
        trainer = self._construct_trainer(new_params=params)
        # not_terminal marks the middle transition as terminal (bootstrap
        # dropped) and the final transition as truncated (bootstrap kept from
        # the supplied next_state).
        not_terminal = torch.tensor([1.0, 0.0, 1.0])
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )._replace(
            next_state=FeatureData(
                float_features=torch.randn(self.batch_size, self.state_dim)
            ),
            not_terminal=not_terminal,
        )
        # value_net is called on state first (baselines), then on next_state.
        state_values = torch.tensor([1.0, 2.0, 3.0])
        next_state_values = torch.tensor([4.0, 5.0, 6.0])
        value_net = trainer.value_net
        assert value_net is not None
        value_net.forward = mock.Mock(side_effect=[state_values, next_state_values])
        # Force the importance ratio to 1 so ppo_loss reduces to -sum(advantage).
        trainer.sampler.log_prob = mock.Mock(return_value=inp.log_prob.detach())

        losses = trainer._trajectory_to_losses(inp)

        # A_t = r_t + gamma * not_terminal_t * V(s_{t+1}) - V(s_t).
        rewards = torch.clamp(inp.reward.detach(), max=trainer.reward_clip)
        td_target = rewards + trainer.gamma * not_terminal * next_state_values
        expected_advantage = td_target - state_values
        self.assertTrue(torch.allclose(losses["ppo_loss"], -expected_advantage.sum()))
        # The critic is trained toward the same bootstrapped TD target, not the
        # (biased) truncated Monte-Carlo return.
        expected_value_loss = torch.nn.functional.mse_loss(
            state_values, td_target, reduction="sum"
        )
        self.assertTrue(torch.allclose(losses["value_net_loss"], expected_value_loss))

    def test_ppo_loss_uses_per_timestep_clip(self) -> None:
        params = self._params(actions=self.actions, normalize=False)
        trainer = self._construct_trainer(new_params=params, use_value_net=False)
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )._replace(log_prob=torch.tensor([-1.0, 0.0, 1.0]))
        # Deterministic importance ratios spanning inside/outside the clip band:
        # ratio = exp(target - log_prob) = exp([1, 0, -1]).
        target_propensity = torch.zeros(self.batch_size)
        trainer.sampler.log_prob = mock.Mock(return_value=target_propensity)

        losses = trainer._trajectory_to_losses(inp)

        advantage = discounted_returns(
            torch.clamp(inp.reward.detach(), max=trainer.reward_clip).clone(),
            trainer.gamma,
        ).float()
        ratio = torch.exp(target_propensity - inp.log_prob.detach()).float()
        clipped = torch.clamp(ratio, 1 - trainer.ppo_epsilon, 1 + trainer.ppo_epsilon)
        # Sum of per-timestep mins, NOT min of the two summed dot-products.
        expected = -torch.min(advantage * ratio, advantage * clipped).sum()
        self.assertTrue(torch.allclose(losses["ppo_loss"], expected))

    def test_rejects_column_vector_rewards_and_log_probs(self) -> None:
        params = self._params(actions=self.actions, normalize=False)
        trainer = self._construct_trainer(new_params=params, use_value_net=False)
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )

        with self.assertRaises(AssertionError):
            trainer._trajectory_to_losses(
                inp._replace(reward=inp.reward.reshape(-1, 1))
            )

        with self.assertRaises(AssertionError):
            trainer._trajectory_to_losses(
                inp._replace(log_prob=inp.log_prob.reshape(-1, 1))
            )

    def test_value_net_loss_sums_over_timesteps(self) -> None:
        params = self._params(actions=self.actions, normalize=False)
        trainer = self._construct_trainer(new_params=params)
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        values = torch.tensor([1.0, 2.0, 3.0])
        value_net = trainer.value_net
        assert value_net is not None
        value_net.forward = mock.Mock(return_value=values)

        losses = trainer._trajectory_to_losses(inp)

        returns = discounted_returns(
            torch.clamp(inp.reward.detach(), max=trainer.reward_clip).clone(),
            trainer.gamma,
        )
        expected_value_loss = torch.nn.functional.mse_loss(
            values, returns, reduction="sum"
        )
        self.assertTrue(torch.allclose(losses["value_net_loss"], expected_value_loss))

    def test_td_error_advantage_next_state_without_not_terminal(self) -> None:
        params = self._params(
            actions=self.actions, normalize=False, td_error_advantage=True
        )
        trainer = self._construct_trainer(new_params=params)
        # next_state present but not_terminal absent: per the contract this is a
        # complete episode, so the final step must NOT bootstrap from V(s').
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )._replace(
            next_state=FeatureData(
                float_features=torch.randn(self.batch_size, self.state_dim)
            ),
        )
        state_values = torch.tensor([1.0, 2.0, 3.0])
        next_state_values = torch.tensor([4.0, 5.0, 6.0])
        value_net = trainer.value_net
        assert value_net is not None
        value_net.forward = mock.Mock(side_effect=[state_values, next_state_values])
        trainer.sampler.log_prob = mock.Mock(return_value=inp.log_prob.detach())

        losses = trainer._trajectory_to_losses(inp)

        # Default mask [1, 1, 0]: interior steps bootstrap, final step terminal.
        rewards = torch.clamp(inp.reward.detach(), max=trainer.reward_clip)
        default_mask = torch.tensor([1.0, 1.0, 0.0])
        td_target = rewards + trainer.gamma * default_mask * next_state_values
        expected_advantage = td_target - state_values
        self.assertTrue(torch.allclose(losses["ppo_loss"], -expected_advantage.sum()))

    def test_entropy_is_summed_over_timesteps(self) -> None:
        params = self._params(actions=self.actions, normalize=False, entropy_weight=0.5)
        trainer = self._construct_trainer(new_params=params, use_value_net=False)
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        # ratio == 1 so the surrogate reduces to -sum(advantage).
        trainer.sampler.log_prob = mock.Mock(return_value=inp.log_prob.detach())
        assert isinstance(trainer.sampler, SoftmaxActionSampler)
        trainer.sampler.entropy = mock.Mock(return_value=torch.tensor(2.0))

        losses = trainer._trajectory_to_losses(inp)

        advantage = discounted_returns(
            torch.clamp(inp.reward.detach(), max=trainer.reward_clip).clone(),
            trainer.gamma,
        ).float()
        # Entropy (a per-step mean) is scaled by the number of steps so it is
        # summed over the trajectory, consistent with the summed surrogate.
        expected = -advantage.sum() - trainer.entropy_weight * 2.0 * self.batch_size
        self.assertTrue(torch.allclose(losses["ppo_loss"], expected))

    def test_invalid_loop_params_rejected(self) -> None:
        for bad in (
            {"update_freq": 0},
            {"update_epochs": 0},
            {"ppo_batch_size": 0},
        ):
            params = self._params(actions=self.actions, normalize=False, **bad)
            with self.assertRaises(AssertionError):
                self._construct_trainer(new_params=params)

    def test_eval_metrics_bootstrap_truncated_trajectory(self) -> None:
        params = self._params(
            actions=self.actions, normalize=False, td_error_advantage=True
        )
        trainer = self._construct_trainer(new_params=params)
        reward = torch.tensor([1.0, 2.0, 3.0])
        not_terminal = torch.tensor([1.0, 1.0, 1.0])
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )._replace(
            reward=reward,
            log_prob=torch.zeros(self.batch_size),
            next_state=FeatureData(
                float_features=torch.randn(self.batch_size, self.state_dim)
            ),
            not_terminal=not_terminal,
        )
        trainer.sampler.log_prob = mock.Mock(return_value=torch.zeros(self.batch_size))
        value_net = trainer.value_net
        assert value_net is not None
        value_net.forward = mock.Mock(return_value=torch.tensor([4.0, 5.0, 6.0]))

        metrics = trainer._eval_metrics([inp])

        final_bootstrap = 6.0
        expected_returns = torch.empty_like(reward)
        running = torch.tensor(final_bootstrap)
        for i in range(self.batch_size - 1, -1, -1):
            running = reward[i] + trainer.gamma * not_terminal[i] * running
            expected_returns[i] = running
        self.assertTrue(
            torch.allclose(metrics["Training/ips_value"], expected_returns.mean())
        )

    def test_configure_optimizers(self) -> None:
        # Ordering is value then policy
        trainer = self._construct_trainer()
        optimizers = trainer.configure_optimizers()
        self.assertTrue(
            torch.all(
                torch.isclose(
                    optimizers[0]["optimizer"].param_groups[0]["params"][0],
                    # pyrefly: ignore [missing-attribute]
                    list(trainer.value_net.fc.dnn[0].parameters())[0],
                )
            )
        )
        self.assertTrue(
            torch.all(
                torch.isclose(
                    optimizers[1]["optimizer"].param_groups[0]["params"][0],
                    # pyrefly: ignore [missing-attribute]
                    list(trainer.scorer.shared_network.fc.dnn[0].parameters())[0],
                )
            )
        )

    def test_get_optimizers(self) -> None:
        # ordering covered in test_configure_optimizers
        trainer = self._construct_trainer()
        optimizers = trainer.get_optimizers()
        self.assertIsNotNone(optimizers[0])
        trainer = self._construct_trainer(use_value_net=False)
        optimizers = trainer.get_optimizers()
        self.assertIsNone(optimizers[0])

    def test_training_step(self) -> None:
        trainer = self._construct_trainer()
        inp = defaultdict(lambda: torch.ones(1, 5))
        trainer.update_model = mock.Mock()
        trainer.training_step(inp, batch_idx=1)
        trainer.update_model.assert_called_with()
        trainer.update_freq = 10
        trainer.update_model = mock.Mock()
        trainer.training_step(inp, batch_idx=1)
        trainer.update_model.assert_not_called()

    def test_update_model(self) -> None:
        trainer = self._construct_trainer()
        # can't update empty model
        with self.assertRaises(AssertionError):
            trainer.update_model()
        # _update_model called with permutation of traj_buffer contents update_epoch # times
        trainer = self._construct_trainer(
            new_params=self._params(
                ppo_batch_size=1,
                update_epochs=2,
                update_freq=2,
                normalize=False,
            )
        )
        trainer.traj_buffer = [1, 2]
        trainer._update_model = mock.Mock()
        trainer.update_model()
        calls = [mock.call([1]), mock.call([2]), mock.call([1]), mock.call([2])]
        trainer._update_model.assert_has_calls(calls, any_order=True)
        # trainer empties buffer
        self.assertEqual(trainer.traj_buffer, [])

        # _update_model
        trainer = self._construct_trainer()
        value_net_opt_mock = mock.Mock()
        ppo_opt_mock = mock.Mock()
        trainer.get_optimizers = mock.Mock(
            return_value=[value_net_opt_mock, ppo_opt_mock]
        )
        trainer._trajectory_to_losses = mock.Mock(
            side_effect=[
                {"ppo_loss": torch.tensor(1), "value_net_loss": torch.tensor(2)},
                {"ppo_loss": torch.tensor(3), "value_net_loss": torch.tensor(4)},
            ]
        )
        trainer.manual_backward = mock.Mock()
        inp1 = PolicyGradientInput.input_prototype(
            batch_size=1, action_dim=1, state_dim=1
        )
        inp2 = PolicyGradientInput.input_prototype(
            batch_size=1, action_dim=1, state_dim=1
        )

        trainer._update_model([inp1, inp2])

        trainer._trajectory_to_losses.assert_has_calls(
            [mock.call(inp1), mock.call(inp2)]
        )
        value_net_opt_mock.zero_grad.assert_called()
        value_net_opt_mock.step.assert_called()

        ppo_opt_mock.zero_grad.assert_called()
        ppo_opt_mock.step.assert_called()

        trainer.manual_backward.assert_has_calls(
            [mock.call(torch.tensor(6)), mock.call(torch.tensor(4))]
        )
