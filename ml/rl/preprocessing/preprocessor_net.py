from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np

from caffe2.python import workspace
import caffe2.proto.caffe2_pb2 as caffe2_pb2

from ml.rl.preprocessing.normalization import \
    BOX_COX_MIN_VALUE, MISSING_VALUE
from ml.rl.preprocessing import identify_types


class PreprocessorNet:
    ZERO = 'ZERO'
    MISSING = 'MISSING_VALUE'
    MISSING_U = 'MISSING_VALUE_U'
    MISSING_L = 'MISSING_VALUE_L'

    def __init__(self, net):
        self._net = net
        workspace.FeedBlob(self.ZERO, np.array([0], dtype=np.float32))
        workspace.FeedBlob(
            self.MISSING, np.array([MISSING_VALUE], dtype=np.float32)
        )
        workspace.FeedBlob(
            self.MISSING_U, np.array([MISSING_VALUE + 1e-4], dtype=np.float32)
        )
        workspace.FeedBlob(
            self.MISSING_L, np.array([MISSING_VALUE - 1e-4], dtype=np.float32)
        )
        self.parameters = [
            self.ZERO, self.MISSING, self.MISSING_L, self.MISSING_U
        ]

    def preprocess_blob(self, blob, normalization_parameters):
        is_empty_u = blob + "__isempty_u"
        is_empty_l = blob + "__isempty_l"
        is_empty = blob + "__isempty"
        self._net.GT([blob, self.MISSING_L], [is_empty_l], broadcast=1)
        self._net.LT([blob, self.MISSING_U], [is_empty_u], broadcast=1)
        self._net.And([is_empty_l, is_empty_u], [is_empty])
        parameters = []
        if normalization_parameters.feature_type == identify_types.BINARY:
            is_gt_zero = blob + "__is_gt_zero"
            is_lt_zero = blob + "__is_lt_zero"
            self._net.GT([blob, self.ZERO], [is_gt_zero], broadcast=1)
            self._net.LT([blob, self.ZERO], [is_lt_zero], broadcast=1)
            bool_blob = blob + "__bool"
            self._net.Or([is_gt_zero, is_lt_zero], [bool_blob])
            self._net.Cast([bool_blob], [blob], to=caffe2_pb2.TensorProto.FLOAT)
        elif normalization_parameters.feature_type == identify_types.PROBABILITY:
            self._net.Clip([blob], [blob], minimum=0.01, maximum=0.99)
            self._net.Logit([blob], [blob])
        else:
            if normalization_parameters.boxcox_lambda is not None:
                boxcox_shift = '{}__boxcox_shift'.format(blob)
                workspace.FeedBlob(
                    boxcox_shift,
                    np.array(
                        normalization_parameters.boxcox_shift, dtype=np.float32
                    )
                )
                boxcox_lambda = '{}__boxcox_lambda'.format(blob)
                workspace.FeedBlob(
                    boxcox_lambda,
                    np.array(
                        normalization_parameters.boxcox_lambda,
                        dtype=np.float32
                    )
                )
                self._net.Sub([blob, boxcox_shift], [blob], broadcast=1, axis=0)
                self._net.Clip([blob], [blob], minimum=BOX_COX_MIN_VALUE)
                self._net.BatchBoxCox([blob, boxcox_lambda, self.ZERO], [blob])
                parameters = [boxcox_lambda, boxcox_shift]

            mean = '{}__preprocess_mean'.format(blob)
            workspace.FeedBlob(
                mean, np.array(normalization_parameters.mean, dtype=np.float32)
            )
            stddev = '{}__preprocess_stddev'.format(blob)
            workspace.FeedBlob(
                stddev,
                np.array(normalization_parameters.stddev, dtype=np.float32)
            )
            self._net.Sub([blob, mean], [blob], broadcast=1, axis=0)
            self._net.Div([blob, stddev], [blob], broadcast=1, axis=0)
            parameters = parameters + [mean, stddev]

        zeros = "zeros"
        self._net.ConstantFill([blob], [zeros], value=0.)
        output_blob = blob + "_preprocessed"
        self._net.Where([is_empty, zeros, blob], [output_blob])
        self._net.ConstantFill([blob], [blob], value=MISSING_VALUE)

        return output_blob, parameters
