"""
How to use:
    buck test reagent:training_tests -- TestDeepRepresentLinUCB
"""

import unittest

import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler

from reagent.models.deep_represent_linucb import DeepRepresentLinearRegressionUCB
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.training.cb.deep_represent_linucb_trainer import DeepRepresentLinUCBTrainer
from reagent.training.parameters import DeepRepresentLinUCBTrainerParameters


class TestDeepRepresentLinUCB(unittest.TestCase):
    """
    This tests the trainer of DeepRepresentLinUCB.
    """

    def setUp(self):

        self.params = DeepRepresentLinUCBTrainerParameters(
            lr=1e-1, loss_type="cross_entropy"
        )

        input_dim = 100
        sizes = [20]
        linucb_inp_dim = 5
        activations = ["relu", "relu"]

        customized_layers = FullyConnectedNetwork(
            [input_dim] + sizes + [linucb_inp_dim],
            activations,
            use_batch_norm=False,
            dropout_ratio=0.0,
            normalize_output=False,
            use_layer_norm=False,
        )
        policy_network = DeepRepresentLinearRegressionUCB(
            input_dim=input_dim,
            sizes=sizes + [linucb_inp_dim],
            activations=activations,
            mlp_layers=customized_layers,
            output_activation="sigmoid",
        )

        self.policy = Policy(scorer=policy_network, sampler=GreedyActionSampler())
        self.trainer = DeepRepresentLinUCBTrainer(self.policy, **self.params.asdict())
        self.batch = CBInput(
            context_arm_features=torch.rand(2, 2, input_dim),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[0.3], [0.1]]),
        )  # random Gaussian features

    def test_linucb_training_step(self):
        self.trainer.training_step(self.batch, 0)
        assert len(self.batch.action) == len(self.batch.reward)
        assert len(self.batch.action) == self.batch.context_arm_features.shape[0]

        loss = self.trainer.training_step(batch=self.batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.size(), ())
