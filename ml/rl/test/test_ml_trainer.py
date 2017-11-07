from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import numpy as np
from scipy import stats
import unittest

from caffe2.python import workspace

from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.ml_trainer_extension import \
    MLTrainerIP  # , MLTrainerExt
from ml.rl.thrift.core.ttypes import TrainingParameters


class TestMLTrainer(unittest.TestCase):
    def test_input_validation(self):
        with self.assertRaises(Exception):
            # layers and activations sizes incompatible
            MLTrainer(
                "Test model",
                TrainingParameters(
                    [2, 1], ['linear', 'relu'], 100, 0.001, 'ADAM'
                )
            )

        with self.assertRaises(Exception):
            # All values in layers should be positive integers
            MLTrainer(
                "Test model",
                TrainingParameters([-1, 1], ['linear'], 100, 0.001, 'ADAM')
            )
            MLTrainer(
                "Test model",
                TrainingParameters([1.3, 1], ['linear'], 100, 0.001, 'ADAM')
            )

    def test_linear_regression_adam(self):
        np.random.seed(0)
        random.seed(1)
        num_features = 4
        num_training_samples = 100
        num_outputs = 1  # change if to test mutliple output
        input_distribution = stats.norm()
        training_inputs = input_distribution.rvs(
            size=(num_training_samples, num_features)
        ).astype(np.float32)
        weights = [[-1, 0.5, 2, 3]]
        if num_outputs == 2:
            weights += [[2, -0.5, 4, 1]]
        weights = np.array(weights).transpose()
        noise = np.random.normal(size=(num_training_samples, num_outputs)) * 0.1
        training_outputs = np.dot(training_inputs, weights) + noise
        training_outputs = training_outputs.astype(np.float32)

        trainer = MLTrainer(
            "Linear Regression",
            TrainingParameters(
                layers=[num_features, num_outputs],
                activations=['linear'],
                minibatch_size=100,
                learning_rate=0.1,
                optimizer='ADAM'
            )
        )
        for _ in range(10000):
            trainer.train(training_inputs, training_outputs)

        trained_weights = np.concatenate(
            [workspace.FetchBlob(w) for w in trainer.weights], axis=0
        ).transpose()
        weights_diff = trained_weights - weights

        self.assertTrue(np.linalg.norm(weights_diff) < 0.05)

    def test_linear_regression_sgd(self):
        np.random.seed(0)
        random.seed(1)
        num_features = 4
        num_training_samples = 100
        num_outputs = 1  # change if to test mutliple output
        input_distribution = stats.norm()
        training_inputs = input_distribution.rvs(
            size=(num_training_samples, num_features)
        ).astype(np.float32)
        weights = [[-1, 0.5, 2, 3]]
        if num_outputs == 2:
            weights += [[2, -0.5, 4, 1]]
        weights = np.array(weights).transpose()
        noise = np.random.normal(size=(num_training_samples, num_outputs)) * 0.1
        training_outputs = np.dot(training_inputs, weights) + noise
        training_outputs = training_outputs.astype(np.float32)
        gamma = 0.9999

        trainer = MLTrainer(
            "Linear Regression",
            TrainingParameters(
                layers=[num_features, num_outputs],
                activations=['linear'],
                minibatch_size=100,
                learning_rate=0.001,
                optimizer='SGD',
                lr_policy='step',
                gamma=gamma,
            )
        )
        for i in range(10000):
            trainer.train(training_inputs, training_outputs)
            curr_neg_LR = np.asscalar(
                workspace.FetchBlob('SgdOptimizer_0_lr_cpu')
            )
            expected_neg_LR = -1.0 * trainer.learning_rate * np.power(gamma, i)

            self.assertTrue(np.isclose(curr_neg_LR, expected_neg_LR, rtol=0.1))

        trained_weights = np.concatenate(
            [workspace.FetchBlob(w) for w in trainer.weights], axis=0
        ).transpose()
        weights_diff = trained_weights - weights
        self.assertTrue(np.linalg.norm(weights_diff) < 0.05)

    def test_identity(self):
        # this is to test corner case of nn w.o. weights, just identity
        # might be useful when reduce from actor-critic, also normalizations
        np.random.seed(0)
        random.seed(1)
        num_features = 4
        num_training_samples = 100
        input_distribution = stats.norm()
        training_inputs = input_distribution.rvs(
            size=(num_training_samples, num_features)
        ).astype(np.float32)
        training_outputs = training_inputs.astype(np.float32)
        trainer = MLTrainer(
            "Identity",
            TrainingParameters(
                layers=[num_features],
                activations=[],
                minibatch_size=100,
                learning_rate=0.001
            )
        )
        identity_output = trainer.score(training_inputs)
        for _ in range(100):
            trainer.train(training_inputs, training_outputs)
        identity_output = trainer.score(training_inputs)
        self.assertTrue(
            np.linalg.norm(identity_output - training_outputs) < 0.01
        )

    def test_scaledoutput_regression(self):
        # this is testing if scaled/inner-product in final layer,
        #  using same case but transformed to (wx+b) * (1/y) = 1,
        #  in this test case, learned weight should be regularized on columns
        np.random.seed(0)
        random.seed(1)
        num_features = 4
        num_training_samples = 100
        num_outputs = 2
        input_distribution = stats.norm()
        training_inputs = input_distribution.rvs(
            size=(num_training_samples, num_features)
        ).astype(np.float32)
        # weights = np.array([[-1, 0.5, 2, 3], [2, -0.5, 4, 1]]).transpose()
        weights = [[-1, 0.5, 2, 3]]
        if num_outputs == 2:
            weights += [[2, -0.5, 4, 1]]
        weights = np.array(weights).transpose()

        noise = np.random.normal(size=(num_training_samples,
                                       num_outputs)) * 0.01
        training_outputs = np.dot(training_inputs, weights) + noise
        training_outputs = training_outputs.astype(np.float32)

        trainer = MLTrainerIP(
            "Linear Regression",
            TrainingParameters(
                layers=[num_features, num_outputs],
                activations=['linear'],
                minibatch_size=100,
                learning_rate=0.1
            ),
            scaled_output=True
        )
        training_labels = np.ones((training_inputs.shape[0],
                                   1)).astype(np.float32)
        training_scale = np.where(
            training_outputs == 0, 0, 1.0 / (num_outputs * training_outputs)
        )

        for _ in range(10000):
            trainer.train_wexternal(
                training_inputs, training_labels, training_scale
            )

        trained_weights = np.concatenate(
            [workspace.FetchBlob(w) for w in trainer.weights], axis=0
        ).transpose()

        # regularized after training to keep same scale as given weights
        scale_trained_weights = np.copy(trained_weights)
        for i in range(trained_weights.shape[1]):
            scaling = 1.0 / trained_weights[0, i] * weights[0, i]
            scale_trained_weights[:, i] = trained_weights[:, i] * scaling

        weights_diff = scale_trained_weights - weights

        self.assertTrue(np.linalg.norm(weights_diff) < 0.05)
