#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python.predictor.predictor_exporter import PredictorExportMeta
from caffe2.python import model_helper
from caffe2.python import workspace

from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.training.rl_predictor import RLPredictor

import logging
logger = logging.getLogger(__name__)


class ContinuousActionDQNPredictor(RLPredictor):
    def predict(self, states, actions):
        """ Returns values for each state/action pair
        :param states states as list of feature -> value dict
        :param actions actions as list of feature -> value dict
        """

        examples = []
        for i in range(len(states)):
            examples.append({**states[i], **actions[i]})
        return RLPredictor.predict(self, examples)

    def get_predictor_export_meta(self):
        return PredictorExportMeta(
            self._net, self._parameters, self._input_blobs, self._output_blobs
        )

    @classmethod
    def export(
        cls,
        trainer,
        state_normalization_parameters,
        action_normalization_parameters,
    ):
        """ Creates ContinuousActionDQNPredictor from a list of action trainers

        :param trainer ContinuousActionDQNPredictor
        :param state_features list of state feature names
        :param action_features list of action feature names
        """
        # ensure state and action IDs have no intersection
        assert (
            len(
                set(state_normalization_parameters.keys()) &
                set(action_normalization_parameters.keys())
            ) == 0
        )

        model = model_helper.ModelHelper(name="predictor")
        net = model.net

        workspace.FeedBlob(
            'input/float_features.lengths', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.keys', np.zeros(1, dtype=np.int32)
        )
        workspace.FeedBlob(
            'input/float_features.values', np.zeros(1, dtype=np.float32)
        )

        preprocessor = PreprocessorNet(net, True)
        parameters = []
        parameters.extend(preprocessor.parameters)
        state_normalized_dense_matrix, new_parameters = \
            preprocessor.normalize_sparse_matrix(
                'input/float_features.lengths',
                'input/float_features.keys',
                'input/float_features.values',
                state_normalization_parameters,
                'state_norm',
            )
        parameters.extend(new_parameters)
        action_normalized_dense_matrix, new_parameters = \
            preprocessor.normalize_sparse_matrix(
                'input/float_features.lengths',
                'input/float_features.keys',
                'input/float_features.values',
                action_normalization_parameters,
                'action_norm',
            )
        parameters.extend(new_parameters)
        state_action_normalized = 'state_action_normalized'
        state_action_normalized_dim = 'state_action_normalized_dim'
        net.Concat(
            [state_normalized_dense_matrix, action_normalized_dense_matrix],
            [state_action_normalized, state_action_normalized_dim],
            axis=1
        )
        new_parameters = RLPredictor._forward_pass(
            model,
            trainer,
            state_action_normalized,
            ['Q'],
        )
        parameters.extend(new_parameters)

        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net)
        return ContinuousActionDQNPredictor(net, parameters)
