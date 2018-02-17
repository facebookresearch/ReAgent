from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python.predictor.predictor_exporter import PredictorExportMeta
from caffe2.python import model_helper
from caffe2.python import workspace

from ml.rl.training.rl_predictor import RLPredictor
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet

import logging
logger = logging.getLogger(__name__)


class DiscreteActionPredictor(RLPredictor):
    def get_predictor_export_meta(self):
        return PredictorExportMeta(
            self._net, self._parameters, self._input_blobs, self._output_blobs
        )

    @classmethod
    def export(cls, trainer, actions, normalization_parameters):
        """ Creates DiscreteActionPredictor from a list of action trainers

        :param trainer DiscreteActionTrainer
        :param features list of state feature names
        :param actions list of action names
        """

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
        normalized_dense_matrix, new_parameters = preprocessor.normalize_sparse_matrix(
            'input/float_features.lengths',
            'input/float_features.keys',
            'input/float_features.values',
            normalization_parameters,
            'state_norm',
        )
        parameters.extend(new_parameters)

        new_parameters = RLPredictor._forward_pass(
            model,
            trainer,
            normalized_dense_matrix,
            actions,
        )
        parameters.extend(new_parameters)

        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net)
        return DiscreteActionPredictor(net, parameters)
