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
from ml.rl.thrift.core.ttypes import TrainingParameters
from ml.rl.training.ml_trainer_extension import MLTrainerIP


def gen_training_data(
    num_features, num_training_samples, num_outputs, noise_scale=0.1,
):
    np.random.seed(0)
    random.seed(1)
    input_distribution = stats.norm()
    training_inputs = input_distribution.rvs(
        size=(num_training_samples, num_features)
    ).astype(np.float32)
    weights = np.random.normal(
        size=(num_outputs, num_features)
    ).astype(np.float32).transpose()
    noise = np.multiply(
        np.random.normal(size=(num_training_samples, num_outputs)), noise_scale
    )
    training_outputs = (
        np.dot(training_inputs, weights) + noise
    ).astype(np.float32)

    return training_inputs, training_outputs, weights, input_distribution


def gen_training_and_test_data(
    num_features,
    num_training_samples,
    num_test_datapoints,
    num_outputs,
    noise_scale=0.1
):
    training_inputs, training_outputs, weights, input_distribution = (
        gen_training_data(
            num_features, num_training_samples, num_outputs, noise_scale
        )
    )

    test_inputs = input_distribution.rvs(
        size=(num_test_datapoints, num_features)
    ).astype(np.float32)
    test_outputs = np.dot(test_inputs, weights).astype(np.float32)
    return training_inputs, training_outputs, test_inputs, test_outputs, weights


def _train(
    trainer,
    num_features,
    num_training_samples,
    num_test_datapoints,
    num_outputs,
    num_training_iterations
):
    training_inputs, training_outputs, test_inputs, test_outputs, weights = (
        gen_training_and_test_data(
            num_features, num_training_samples, num_test_datapoints, num_outputs
        )
    )
    for _ in range(num_training_iterations):
        trainer.train_batch(training_inputs, training_outputs)

    return test_inputs, test_outputs, weights


def get_prediction_dist(
    trainer,
    num_outputs=1,
    num_features=4,
    num_training_samples=100,
    num_test_datapoints=10,
    num_training_iterations=10000,
):
    test_inputs, test_outputs, _ = _train(
        trainer,
        num_features,
        num_training_samples,
        num_test_datapoints,
        num_outputs,
        num_training_iterations
    )

    predictions = trainer.score(test_inputs)
    dist = np.linalg.norm(test_outputs - predictions)
    return dist


def get_weight_dist(
        trainer,
        num_outputs=1,
        num_features=4,
        num_training_samples=100,
        num_test_datapoints=10,
        num_training_iterations=10000,
):
    _, _, weights = _train(
        trainer,
        num_features,
        num_training_samples,
        num_test_datapoints,
        num_outputs,
        num_training_iterations
    )

    trained_weights = np.concatenate(
        [workspace.FetchBlob(w) for w in trainer.weights], axis=0
    ).transpose()
    dist = np.linalg.norm(trained_weights - weights)
    return dist


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
                learning_rate=0.1,
                optimizer='ADAM'
            )
        )
        identity_output = trainer.score(training_inputs)
        for _ in range(100):
            trainer.train_batch(training_inputs, training_outputs)
        identity_output = trainer.score(training_inputs)
        self.assertTrue(
            np.linalg.norm(identity_output - training_outputs) < 0.01
        )

    def test_sgd_dropout_predictions(self):
        num_features = 4
        num_outputs = 1

        trainer = MLTrainer(
            "Linear Regression",
            TrainingParameters(
                layers=[num_features, num_outputs],
                activations=['linear'],
                minibatch_size=100,
                learning_rate=0.001,
                optimizer='SGD',
                gamma=0.9999,
                lr_policy='step',
                dropout_ratio=0.05
            )
        )

        dist = get_prediction_dist(
            trainer, num_features=num_features, num_outputs=num_outputs
        )
        self.assertLess(dist, 1.0)

    def test_sgd_weights(self):
        num_features = 4
        num_outputs = 1

        trainer = MLTrainer(
            "Linear Regression",
            TrainingParameters(
                layers=[num_features, num_outputs],
                activations=['linear'],
                minibatch_size=100,
                learning_rate=0.001,
                optimizer='SGD',
                gamma=0.9999,
                lr_policy='step'
            )
        )

        dist = get_weight_dist(
            trainer, num_features=num_features, num_outputs=num_outputs
        )

        self.assertLess(dist, 0.1)

    def test_adam_dropout_predictions(self):
        num_features = 4
        num_outputs = 1

        trainer = MLTrainer(
            "Linear Regression",
            TrainingParameters(
                layers=[num_features, num_outputs],
                activations=['linear'],
                minibatch_size=100,
                learning_rate=0.1,
                optimizer='ADAM',
                dropout_ratio=0.1
            )
        )

        dist = get_prediction_dist(
            trainer, num_features=num_features, num_outputs=num_outputs
        )

        self.assertLess(dist, 2.0)

    def test_adam_weights(self):
        num_features = 4
        num_outputs = 1

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

        dist = get_weight_dist(
            trainer, num_features=num_features, num_outputs=num_outputs
        )

        self.assertLess(dist, 0.1)

    def test_scaledoutput_regression_weights(self):
        # this is testing if scaled/inner-product in final layer,
        #  using same case but transformed to (wx+b) * (1/y) = 1,
        #  in this test case, learned weight should be regularized on columns
        num_features = 4
        num_training_samples = 100
        num_outputs = 2

        training_inputs, training_outputs, weights, _ = (
            gen_training_data(
                num_features, num_training_samples, num_outputs, 0.01
            )
        )

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

        training_labels = np.ones(
            (training_inputs.shape[0], 1)
        ).astype(np.float32)
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

        dist = np.linalg.norm(scale_trained_weights - weights)
        self.assertLess(dist, 0.1)
