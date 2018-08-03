#!/usr/bin/env python3

import os
import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np
from caffe2.python import core, workspace
from caffe2.python.predictor.predictor_exporter import PredictorExportMeta, save_to_db
from ml.rl.caffe_utils import C2
from ml.rl.evaluation.policy_evaluator import (
    PolicyEvaluator,
    Slate,
    save_sum_deterministic_policy,
)
from ml.rl.thrift.eval.ttypes import (
    PolicyEvaluatorParameters,
    ValueInputModelParameters,
)


class TestPolicyEvaluator(unittest.TestCase):
    def test_no_model_nets_matching_policy(self):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            policy_file = os.path.join(temp_directory_name, "policy.db")
            pep = PolicyEvaluatorParameters()
            pep.propensity_net_path = policy_file
            pep.db_type = "minidb"
            pep.global_value_inputs = {}
            pep.value_input_models = []
            pep.value_input_models.append(ValueInputModelParameters(name="Model1"))
            pep.value_input_models.append(ValueInputModelParameters(name="Model2"))
            save_sum_deterministic_policy(
                ["Model1", "Model2"], policy_file, pep.db_type
            )
            pe = PolicyEvaluator(pep)
            slates = []

            # First slate: new model picks action 0
            slates.append(
                self.flatten_slate(
                    {},
                    [
                        {
                            "Model1": {"Output": 1.0},
                            "Model2": {"Output": 0.0},
                        },  # First example scores
                        {
                            "Model1": {"Output": 0.0},
                            "Model2": {"Output": 0.0},
                        },  # Second example scores
                    ],
                )
            )

            # Second slate: new model picks action 1
            slates.append(
                self.flatten_slate(
                    {},
                    [
                        {
                            "Model1": {"Output": 0.0},
                            "Model2": {"Output": 0.0},
                        },  # First example scores
                        {
                            "Model1": {"Output": 1.0},
                            "Model2": {"Output": 0.0},
                        },  # Second example scores
                    ],
                )
            )

            rewards = np.array([[1.0], [0.5]])

            # Old model picks actions 0 and 1 with 100% probability
            action_selection = np.array([[0], [1]])
            baseline_probabilities = np.array([[1.0], [1.0]])
            value = pe.evaluate_slates(
                slates, action_selection, rewards, baseline_probabilities
            )
            print(value)
            self.assertAlmostEqual(value, 0.75)

    def test_one_model_net_matching_policy(self):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            policy_file = os.path.join(temp_directory_name, "policy.db")
            model1_file = os.path.join(temp_directory_name, "model1.db")
            self._dummy_model_copy("Model1", model1_file)
            pep = PolicyEvaluatorParameters()
            pep.propensity_net_path = policy_file
            pep.db_type = "minidb"
            pep.global_value_inputs = {}
            pep.value_input_models = []
            pep.value_input_models.append(
                ValueInputModelParameters(name="Model1", path=model1_file)
            )
            pep.value_input_models.append(ValueInputModelParameters(name="Model2"))
            save_sum_deterministic_policy(
                ["Model1", "Model2"], policy_file, pep.db_type
            )
            pe = PolicyEvaluator(pep)
            slates = []

            # First slate: new model picks action 0
            slates.append(
                self.flatten_slate(
                    {},
                    [
                        {
                            "Model1": {"Input": 1.0},
                            "Model2": {"Output": 0.0},
                        },  # First example scores
                        {
                            "Model1": {"Input": 0.0},
                            "Model2": {"Output": 0.0},
                        },  # Second example scores
                    ],
                )
            )

            # Second slate: new model picks action 1
            slates.append(
                self.flatten_slate(
                    {},
                    [
                        {
                            "Model1": {"Input": 0.0},
                            "Model2": {"Output": 0.0},
                        },  # First example scores
                        {
                            "Model1": {"Input": 1.0},
                            "Model2": {"Output": 0.0},
                        },  # Second example scores
                    ],
                )
            )

            rewards = np.array([[1.0], [0.5]])

            # Old model picks actions 0 and 1 with 100% probability
            action_selection = np.array([[0], [1]])
            baseline_probabilities = np.array([[1.0], [1.0]])
            value = pe.evaluate_slates(
                slates, action_selection, rewards, baseline_probabilities
            )
            print(value)
            self.assertAlmostEqual(value, 0.75)

    def flatten_slate(
        self,
        policy_features: Dict[str, Any],
        model_features_per_example: List[Dict[str, Dict[str, Any]]],
    ) -> Slate:
        """
        This function takes in a slate of choices, where each choice is a list
            of either (1) features or (2) model outputs.  The function then
            collates this slate by model, so the output is a list of size N,
            where N is the number of models.  Each element is either a set
            of feature lists with one element per example, or a set of ouptut
            scores.
        """
        all_model_features: Dict[str, Dict[str, List[Any]]] = {}
        for example in model_features_per_example:
            for model_name, new_model_features in example.items():
                if model_name not in all_model_features:
                    all_model_features[model_name] = {}
                model_features = all_model_features[model_name]
                for feature_id, feature_value in new_model_features.items():
                    if feature_id not in model_features:
                        model_features[feature_id] = [feature_value]
                    else:
                        model_features[feature_id].append(feature_value)

        all_model_features_numpy: Dict[str, Dict[str, Any]] = {}
        for model_name, model in all_model_features.items():
            all_model_features_numpy[model_name] = {}
            for feature_id in model.keys():
                all_model_features_numpy[model_name][
                    feature_id
                ] = np.array(  # type: ignore
                    model[feature_id], dtype=np.float32  # type: ignore
                ).reshape(
                    -1, 1
                )

        print(all_model_features_numpy)
        return Slate(policy_features, all_model_features_numpy)

    def _dummy_model_copy(self, model_name, path):
        net = core.Net(model_name)
        C2.set_net(net)
        inp = "Input"
        output = "Output"
        workspace.FeedBlob(inp, np.array([1.0]))
        workspace.FeedBlob(output, np.array([1.0]))
        net.Copy([inp], [output])
        meta = PredictorExportMeta(net, [], [inp], [output])
        save_to_db("minidb", path, meta)
