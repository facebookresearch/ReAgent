from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import six
import sys
import uuid
# @build:deps [
# @/caffe2/caffe2/fb:log_file_db
# @/caffe2/caffe2/python:caffe2_py
# ]

from caffe2.python.predictor.predictor_exporter import \
    save_to_db, load_from_db, prepare_prediction_net
from caffe2.python import workspace
from caffe2.python.predictor_constants import predictor_constants
from caffe2.python.predictor.predictor_py_utils import GetBlobs

import logging
logger = logging.getLogger(__name__)


class RLPredictor(object):
    normalized_input = "PredictorInput"

    def __init__(
        self, net, input_blobs, output_blobs, parameters, workspace_id
    ):
        """

        :param net caffe2 net used for prediction
        :param input_blobs caffe2 blobs used as input
        :param output_blobs caffe2 blobs used as output
        :param parameters caffe2 blobs used as network paramers
        """
        self._net = net
        self._input_blobs = input_blobs
        self._output_blobs = output_blobs
        self._parameters = parameters
        self._workspace_id = workspace_id

    def get_predictor_export_meta(self):
        """
        Returns a PredictorExportMeta object
        """
        pass

    def save(self, db_path, db_type):
        """ Saves network to db

        :param db_path see save_to_db
        :param db_type see save_to_db
        """
        previous_workspace = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace(self._workspace_id)
        meta = self.get_predictor_export_meta()
        for parameter in self._parameters:
            if np.any(np.isnan(workspace.FetchBlob(parameter))):
                logger.info("WARNING: parameter {} is nan".format(parameter))
        try:
            save_to_db(db_type, db_path, meta)
        except Exception:
            logger.info(
                "Warning:failed to save model to db" + sys.exc_info()[0]
            )
        workspace.SwitchWorkspace(previous_workspace)

    @classmethod
    def load(cls, db_path, db_type):
        """ Creates DiscreteActionPredictor by loading from a database

        :param db_path see load_from_db
        :param db_type see load_from_db
        """
        previous_workspace = workspace.CurrentWorkspace()
        workspace_id = str(uuid.uuid4())
        workspace.SwitchWorkspace(workspace_id, True)
        net = prepare_prediction_net(db_path, db_type)
        meta = load_from_db(db_path, db_type)
        inputs = GetBlobs(meta, predictor_constants.INPUTS_BLOB_TYPE)
        outputs = GetBlobs(meta, predictor_constants.OUTPUTS_BLOB_TYPE)
        parameters = GetBlobs(meta, predictor_constants.PARAMETERS_BLOB_TYPE)
        workspace.SwitchWorkspace(previous_workspace)
        return cls(net, inputs, outputs, parameters, workspace_id)

    def analyze(self, named_features):
        print("==================== Model parameters =========================")
        previous_workspace = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace(self._workspace_id)

        for parameter in self._parameters:
            parameter_value = workspace.FetchBlob(parameter)
            print()
            print("Parameter {}:".format(parameter))
            print(parameter_value)
            print()
            print()

        print()
        print("==================== Output ============================")
        for _ in range(3):
            score = self.predict(named_features)
            print(score)
        print()

        print("==================== Input =========================")
        for name, value in six.iteritems(named_features):
            print("Feature {}: {}".format(name, value))

        print()
        print("==================== Normalized Input =========================")
        for name in named_features:
            norm_blob_value = workspace.FetchBlob(name + "_preprocessed")
            print("Normalized Feature {}: {}".format(name, norm_blob_value))

        workspace.SwitchWorkspace(previous_workspace)
