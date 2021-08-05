import unittest
from collections import defaultdict
from unittest import mock

import torch
from reagent.core.types import PolicyGradientInput
from reagent.evaluation.evaluator import get_metrics_to_score
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.models.dueling_q_network import DuelingQNetwork
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.training.parameters import PPOTrainerParameters
from reagent.training.ppo_trainer import PPOTrainer
from reagent.workflow.types import RewardOptions


class TestPPO(unittest.TestCase):
    def setUp(self):
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
        self.params = PPOTrainerParameters(actions=self.actions, normalize=False)
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

        self.value_network = FullyConnectedNetwork(
            layers=[self.state_dim] + self.sizes + [1],
            activations=self.activations + ["linear"],
            use_layer_norm=self.use_layer_norm,
        )

    def _construct_trainer(self, new_params=None, use_value_net=True):
        value_network = self.value_network if use_value_net else None
        params = new_params if new_params else self.params

        trainer = PPOTrainer(
            policy=self.policy, value_net=value_network, **params.asdict()
        )
        trainer.optimizers = mock.Mock(return_value=[0, 0])
        return trainer

    def test_init(self):
        trainer = self._construct_trainer()

        self.assertEqual(
            type(trainer.value_loss_fn), type(torch.nn.MSELoss(reduction="mean"))
        )

        with self.assertRaises(AssertionError):
            new_params = PPOTrainerParameters(ppo_epsilon=-1)
            self._construct_trainer(new_params)

        with self.assertRaises(AssertionError):
            new_params = PPOTrainerParameters(ppo_epsilon=2)
            self._construct_trainer(new_params)

    def test__trajectory_to_losses(self):
        inp = PolicyGradientInput.input_prototype(
            batch_size=self.batch_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        # Normalize + offset clamp min
        params = PPOTrainerParameters(
            actions=["1", "2"], normalize=True, offset_clamp_min=True
        )
        trainer = self._construct_trainer(new_params=params, use_value_net=False)
        losses = trainer._trajectory_to_losses(inp)
        self.assertEqual(len(losses), 1)
        self.assertTrue("ppo_loss" in losses)

        # value net (catch normalize)
        trainer = self._construct_trainer(new_params=params)
        with self.assertRaises(RuntimeError):
            losses = trainer._trajectory_to_losses(inp)

        trainer = self._construct_trainer()
        losses = trainer._trajectory_to_losses(inp)
        self.assertEqual(len(losses), 2)
        self.assertTrue("ppo_loss" in losses and "value_net_loss" in losses)
        # entropy weight should always lower ppo_loss
        trainer.entropy_weight = 1.0
        entropy_losses = trainer._trajectory_to_losses(inp)
        self.assertTrue(entropy_losses["ppo_loss"] < losses["ppo_loss"])

    def test_configure_optimizers(self):
        # Ordering is value then policy
        trainer = self._construct_trainer()
        optimizers = trainer.configure_optimizers()
        self.assertTrue(
            torch.all(
                torch.isclose(
                    optimizers[0]["optimizer"].param_groups[0]["params"][0],
                    list(trainer.value_net.dnn[0].parameters())[0],
                )
            )
        )
        self.assertTrue(
            torch.all(
                torch.isclose(
                    optimizers[1]["optimizer"].param_groups[0]["params"][0],
                    list(trainer.scorer.shared_network.fc.dnn[0].parameters())[0],
                )
            )
        )

    def test_get_optimizers(self):
        # ordering covered in test_configure_optimizers
        trainer = self._construct_trainer()
        optimizers = trainer.get_optimizers()
        self.assertIsNotNone(optimizers[0])
        trainer = self._construct_trainer(use_value_net=False)
        optimizers = trainer.get_optimizers()
        self.assertIsNone(optimizers[0])

    def test_training_step(self):
        trainer = self._construct_trainer()
        inp = defaultdict(lambda: torch.ones(1, 5))
        trainer.update_model = mock.Mock()
        trainer.training_step(inp, batch_idx=1)
        trainer.update_model.assert_called_with()
        trainer.update_freq = 10
        trainer.update_model = mock.Mock()
        trainer.training_step(inp, batch_idx=1)
        trainer.update_model.assert_not_called()

    def test_update_model(self):
        trainer = self._construct_trainer()
        # can't update empty model
        with self.assertRaises(AssertionError):
            trainer.update_model()
        # _update_model called with permutation of traj_buffer contents update_epoch # times
        trainer = self._construct_trainer(
            new_params=PPOTrainerParameters(
                ppo_batch_size=1, update_epochs=2, update_freq=2
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
