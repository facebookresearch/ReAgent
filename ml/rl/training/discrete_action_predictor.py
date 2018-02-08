from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.predictor.predictor_exporter import PredictorExportMeta
from caffe2.python import model_helper
from caffe2.python import workspace

from ml.rl.training.rl_predictor import RLPredictor

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

        normalized_dense_matrix, parameters = RLPredictor._sparse_to_normalized_dense(
            net,
            normalization_parameters,
            'state_norm',
        )

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
