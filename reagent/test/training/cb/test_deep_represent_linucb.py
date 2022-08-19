import unittest

import numpy as np
import numpy.testing as npt
import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler

from reagent.models.deep_represent_linucb import DeepRepresentLinearRegressionUCB
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.training.cb.deep_represent_linucb_trainer import DeepRepresentLinUCBTrainer
from reagent.training.parameters import DeepRepresentLinUCBTrainerParameters


class TestDeepRepresentLinUCB(unittest.TestCase):
    def setUp(self):

        self.params = DeepRepresentLinUCBTrainerParameters()

        raw_input_dim = 100
        sizes = [100]
        linucb_inp_dim = 100
        activations = ["relu"]
        output_activation = "linear"

        customized_layers = FullyConnectedNetwork(
            [raw_input_dim] + sizes + [linucb_inp_dim],
            activations + [output_activation],
            use_batch_norm=False,
            dropout_ratio=0.0,
            normalize_output=False,
            use_layer_norm=False,
        )
        policy_network = DeepRepresentLinearRegressionUCB(
            raw_input_dim=raw_input_dim,
            sizes=sizes,
            linucb_inp_dim=linucb_inp_dim,
            activations=activations,
            mlp_layers=customized_layers,
        )

        self.policy = Policy(scorer=policy_network, sampler=GreedyActionSampler())
        self.trainer = DeepRepresentLinUCBTrainer(self.policy, **self.params.asdict())
        self.batch = CBInput(
            context_arm_features=torch.rand(2, 2, 100),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[1.5], [-2.3]]),
        )  # random Gaussian features where feature_dim=100

    def test_linucb_training_step(self):
        self.trainer.training_step(self.batch, 0)
        assert len(self.batch.action) == len(self.batch.reward)
        assert len(self.batch.action) == self.batch.context_arm_features.shape[0]

        loss_iterations = []
        for _ in range(100):
            # Linucb parameters are updated within training_step manually
            loss = self.trainer.training_step(batch=self.batch, batch_idx=0).item()
            loss_iterations.append(loss)

        npt.assert_allclose(
            np.asarray(loss_iterations[90:]),
            np.zeros(10),
            atol=1e-2,
        )
        return loss_iterations


def test_deep_represent():
    test = TestDeepRepresentLinUCB()
    test.test_linucb_training_step()


"""
How to use:
    buck test reagent:training_tests -- test_deep_represent

    or

    test = TestDeepRepresentLinUCB()
    test.setUp()
    losses = test.test_linucb_training_step()
"""
