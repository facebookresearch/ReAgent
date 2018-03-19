from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tempfile
import unittest
import os

from caffe2.python import core, workspace
from caffe2.python.predictor.predictor_exporter import (
    save_to_db,
    PredictorExportMeta,
)

from ml.rl.thrift.eval.ttypes import (
    PolicyEvaluatorParameters, ValueModelParameters
)

from ml.rl.caffe_utils import C2
from ml.rl.evaluation.policy_evaluator import PolicyEvaluator


class TestPolicyEvaluator(unittest.TestCase):
    def test_no_model_nets_matching_policy(self):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            policy_file = os.path.join(temp_directory_name, 'policy.db')
            self._sum_deterministic_policy(['Model1', 'Model2'], policy_file)
            pep = PolicyEvaluatorParameters()
            pep.propensity_net_path = policy_file
            pep.value_models = []
            pep.value_models.append(ValueModelParameters(name='Model1'))
            pep.value_models.append(ValueModelParameters(name='Model2'))
            pe = PolicyEvaluator(pep, 'minidb')
            slates = []

            # First slate: new model picks action 0
            slates.append(self._generate_numpy_slate([  #
                [1.0, 0.0],  # First example scores
                [0.0, 0.0]  # Second example scores
            ]))

            # Second slate: new model picks action 1
            slates.append(self._generate_numpy_slate([  #
                [0.0, 0.0],  # First example scores
                [1.0, 0.0]  # Second example scores
            ]))

            rewards = np.array([1.0, 0.5])

            # Old model picks actions 0 and 1 with 100% probability
            action_selection = np.array([0, 1])
            baseline_probabilities = np.array([1.0, 1.0])
            value = pe.evaluate_slates(
                slates, action_selection, rewards, baseline_probabilities
            )
            print(value)
            self.assertAlmostEqual(value, 0.75)

    def test_one_model_net_matching_policy(self):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            policy_file = os.path.join(temp_directory_name, 'policy.db')
            self._sum_deterministic_policy(['Model1', 'Model2'], policy_file)
            model1_file = os.path.join(temp_directory_name, 'model1.db')
            self._dummy_model_copy('Model1', model1_file)
            pep = PolicyEvaluatorParameters()
            pep.propensity_net_path = policy_file
            pep.value_models = []
            pep.value_models.append(
                ValueModelParameters(name='Model1', path=model1_file)
            )
            pep.value_models.append(ValueModelParameters(name="Model2"))
            pe = PolicyEvaluator(pep, 'minidb')
            slates = []

            # First slate: new model picks action 0
            slates.append(self._generate_numpy_slate([  #
                [{'Input': 1.0}, 0.0],  # First example scores
                [{'Input': 0.0}, 0.0]  # Second example scores
            ]))

            # Second slate: new model picks action 1
            slates.append(self._generate_numpy_slate([  #
                [{'Input': 0.0}, 0.0],  # First example scores
                [{'Input': 1.0}, 0.0]  # Second example scores
            ]))

            rewards = np.array([1.0, 0.5])

            # Old model picks actions 0 and 1 with 100% probability
            action_selection = np.array([0, 1])
            baseline_probabilities = np.array([1.0, 1.0])
            value = pe.evaluate_slates(
                slates, action_selection, rewards, baseline_probabilities
            )
            print(value)
            self.assertAlmostEqual(value, 0.75)

    def _generate_numpy_slate(
        self,
        slate,
    ):
        """
        This function takes in a slate of choices, where each choice is a list
            of either (1) features or (2) model outputs.  The function then
            collates this slate by model, so the output is a list of size N,
            where N is the number of models.  Each element is either a set
            of feature lists with one element per example, or a set of ouptut
            scores.
        """
        numpy_slate = []
        for model_index in range(len(slate[0])):
            print(slate[0][model_index])
            if isinstance(slate[0][model_index], dict):
                # Model is a net
                model_blobs = {}
                for k in slate[0][model_index].keys():
                    model_blobs[k] = []
                for x in range(len(slate)):
                    for k, v in model_blobs.items():
                        v.append(slate[x][model_index][k])
                for k in model_blobs.keys():
                    model_blobs[k] = np.array(
                        model_blobs[k], dtype=np.float32
                    ).reshape(-1, 1)
                numpy_slate.append(model_blobs)
            else:
                model_scores = []
                for x in range(len(slate)):
                    model_scores.append(slate[x][model_index])
                print(model_scores)
                numpy_slate.append(
                    {
                        'Output':
                            np.array(model_scores, dtype=np.float32)
                            .reshape(-1, 1)
                    }
                )
        return numpy_slate

    def _dummy_model_copy(self, model_name, path):
        net = core.Net(model_name)
        C2.set_net(net)
        inp = 'Input'
        output = 'Output'
        workspace.FeedBlob(inp, np.array([1.0]))
        workspace.FeedBlob(output, np.array([1.0]))
        net.Copy([inp], [output])
        meta = PredictorExportMeta(
            net,
            [],
            [inp],
            [output],
        )
        save_to_db('minidb', path, meta)

    def _sum_deterministic_policy(self, model_names, path):
        net = core.Net('DeterministicPolicy')
        C2.set_net(net)
        output = 'ActionProbabilities'
        workspace.FeedBlob(output, np.array([1.0]))
        model_outputs = []
        for model in model_names:
            model_output = '{}_Output'.format(model)
            workspace.FeedBlob(model_output, np.array([1.0], dtype=np.float32))
            model_outputs.append(model_output)
        max_action = C2.FlattenToVec(
            C2.RowWiseArgMax(C2.Transpose(C2.Sum(*model_outputs)))
        )
        one_blob = C2.NextBlob('one')
        workspace.FeedBlob(one_blob, np.array([1.0], dtype=np.float32))
        C2.net().SparseToDense(
            [
                max_action,
                one_blob,
                model_outputs[0],
            ],
            [output],
        )
        meta = PredictorExportMeta(
            net,
            [one_blob],
            model_outputs,
            [output],
        )
        save_to_db('minidb', path, meta)
