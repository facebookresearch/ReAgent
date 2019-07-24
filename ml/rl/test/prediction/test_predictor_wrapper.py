#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import ml.rl.types as rlt
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.prediction.predictor_wrapper import (
    DiscreteDqnPredictorWrapper,
    DiscreteDqnWithPreprocessor,
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
)
from ml.rl.preprocessing.identify_types import CONTINUOUS
from ml.rl.preprocessing.normalization import NormalizationParameters
from ml.rl.preprocessing.preprocessor import Preprocessor


def _cont_norm():
    return NormalizationParameters(feature_type=CONTINUOUS, mean=0.0, stddev=1.0)


class TestPredictorWrapper(unittest.TestCase):
    def test_discrete_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_dim = 2
        dqn = FullyConnectedDQN(
            state_dim=len(state_normalization_parameters),
            action_dim=action_dim,
            sizes=[16],
            activations=["relu"],
        )
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(dqn, state_preprocessor)
        action_names = ["L", "R"]
        wrapper = DiscreteDqnPredictorWrapper(dqn_with_preprocessor, action_names)
        input_prototype = dqn_with_preprocessor.input_prototype()
        output_action_names, q_values = wrapper(*input_prototype)
        self.assertEqual(action_names, output_action_names)
        self.assertEqual(q_values.shape, (1, 2))

        expected_output = dqn(
            rlt.PreprocessedState.from_tensor(state_preprocessor(*input_prototype[0]))
        ).q_values
        self.assertTrue((expected_output == q_values).all())

    def test_parametric_wrapper(self):
        state_normalization_parameters = {i: _cont_norm() for i in range(1, 5)}
        action_normalization_parameters = {i: _cont_norm() for i in range(5, 9)}
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_preprocessor = Preprocessor(action_normalization_parameters, False)
        dqn = FullyConnectedParametricDQN(
            state_dim=len(state_normalization_parameters),
            action_dim=len(action_normalization_parameters),
            sizes=[16],
            activations=["relu"],
        )
        dqn_with_preprocessor = ParametricDqnWithPreprocessor(
            dqn,
            state_preprocessor=state_preprocessor,
            action_preprocessor=action_preprocessor,
        )
        wrapper = ParametricDqnPredictorWrapper(dqn_with_preprocessor)

        input_prototype = dqn_with_preprocessor.input_prototype()
        output_action_names, q_value = wrapper(*input_prototype)
        self.assertEqual(output_action_names, ["Q"])
        self.assertEqual(q_value.shape, (1, 1))

        expected_output = dqn(
            rlt.PreprocessedStateAction.from_tensors(
                state=state_preprocessor(*input_prototype[0]),
                action=action_preprocessor(*input_prototype[1]),
            )
        ).q_value
        self.assertTrue((expected_output == q_value).all())
