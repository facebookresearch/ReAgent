#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
import six
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
from caffe2.python.predictor.predictor_exporter import (  # type: ignore
    PredictorExportMeta,
    load_from_db,
    prepare_prediction_net,
    save_to_db,
)
from caffe2.python.predictor.predictor_py_utils import GetBlobs, GetNet
from caffe2.python.predictor_constants import predictor_constants
from ml.rl.caffe_utils import C2


logger = logging.getLogger(__name__)


class RLPredictor:
    def __init__(self, net, init_net, parameters, ws=None):
        """
        :param net caffe2 net used for prediction
        :param parameters caffe2 blobs used as network paramers
        """
        self._net = net
        self._init_net = init_net
        self._input_blobs = [
            "input/float_features.lengths",
            "input/float_features.keys",
            "input/float_features.values",
        ]
        self._output_blobs = [
            "output/string_weighted_multi_categorical_features.keys",
            "output/string_weighted_multi_categorical_features.lengths",
            "output/string_weighted_multi_categorical_features.values.keys",
            "output/string_weighted_multi_categorical_features.values.lengths",
            "output/string_weighted_multi_categorical_features.values.values",
        ]
        self._parameters = parameters
        self.ws = ws or workspace

    @property
    def predict_net(self):
        return self._net

    def in_order_dense_to_sparse(self, dense):
        """Convert dense observation to sparse observation assuming in order
        feature ids."""
        sparse = []
        for row in dense:
            sparse.append({str(k): v for k, v in enumerate(row)})
        return sparse

    def predict(self, float_state_features):
        """ Returns values for each state
        :param float_state_features A list of feature -> float value dict examples
        """
        float_state_keys = []
        float_state_values = []
        for example in float_state_features:
            for k, v in example.items():
                float_state_keys.append(k)
                float_state_values.append(v)
        self.ws.FeedBlob(
            "input/float_features.lengths",
            np.array([len(e) for e in float_state_features], dtype=np.int32),
        )
        self.ws.FeedBlob(
            "input/float_features.keys", np.array(float_state_keys, dtype=np.int64)
        )
        self.ws.FeedBlob(
            "input/float_features.values",
            np.array(float_state_values, dtype=np.float32).flatten(),
        )

        self.ws.RunNet(self.predict_net)

        output_lengths = self.ws.FetchBlob(
            "output/string_weighted_multi_categorical_features.values.lengths"
        )
        output_names = self.ws.FetchBlob(
            "output/string_weighted_multi_categorical_features.values.keys"
        )
        output_values = self.ws.FetchBlob(
            "output/string_weighted_multi_categorical_features.values.values"
        )
        assert len(output_lengths) == len(float_state_features), (
            "Invalid number of outputs: "
            + str(len(output_lengths))
            + " != "
            + str(len(float_state_features))
        )

        results = []

        cursor = 0
        for length in output_lengths:
            cursor_begin = cursor
            cursor_end = cursor_begin + length
            cursor = cursor_end

            result = {}
            for x in range(cursor_begin, cursor_end):
                result[output_names[x].decode("utf-8")] = output_values[x]
            results.append(result)

        return results

    def get_predictor_export_meta(self):
        """
        Returns a PredictorExportMeta object
        """
        return PredictorExportMeta(
            self._net,
            self._parameters,
            self._input_blobs,
            self._output_blobs,
            extra_init_net=self._init_net,
        )

    def save(self, db_path, db_type):
        """ Saves network to db

        :param db_path see save_to_db
        :param db_type see save_to_db
        """
        meta = self.get_predictor_export_meta()
        for parameter in self._parameters:
            parameter_data = workspace.FetchBlob(parameter)
            if parameter_data.dtype.kind in {"U", "S", "O"}:
                continue  # Don't bother checking string blobs for nan
            if np.any(np.isnan(parameter_data)):
                logger.info("WARNING: parameter {} is nan".format(parameter))
        save_to_db(db_type, db_path, meta)

    @classmethod
    def load(cls, db_path, db_type):
        """ Creates Predictor by loading from a database

        :param db_path see load_from_db
        :param db_type see load_from_db
        """
        meta = load_from_db(db_path, db_type)
        init_net = GetNet(meta, predictor_constants.PREDICT_INIT_NET_TYPE)
        net = prepare_prediction_net(db_path, db_type)
        parameters = GetBlobs(meta, predictor_constants.PARAMETERS_BLOB_TYPE)
        return cls(net, parameters, init_net)

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

    @classmethod
    def _forward_pass(
        cls, model, trainer, normalized_dense_matrix, actions, qnet_output_blob
    ):
        C2.set_model(model)

        parameters = []
        q_values = "q_values"
        C2.net().Copy([qnet_output_blob], [q_values])

        action_names = C2.NextBlob("action_names")
        parameters.append(action_names)
        workspace.FeedBlob(action_names, np.array(actions))
        action_range = C2.NextBlob("action_range")
        parameters.append(action_range)
        workspace.FeedBlob(action_range, np.array(list(range(len(actions)))))

        output_shape = C2.Shape(q_values)
        output_shape_row_count = C2.Slice(output_shape, starts=[0], ends=[1])

        output_row_shape = C2.Slice(q_values, starts=[0, 0], ends=[-1, 1])

        output_feature_keys = "output/string_weighted_multi_categorical_features.keys"
        workspace.FeedBlob(output_feature_keys, np.zeros(1, dtype=np.int64))
        output_feature_keys_matrix = C2.ConstantFill(
            output_row_shape, value=0, dtype=caffe2_pb2.TensorProto.INT64
        )
        # Note: sometimes we need to use an explicit output name, so we call
        #  C2.net().Fn(...)
        C2.net().FlattenToVec([output_feature_keys_matrix], [output_feature_keys])

        output_feature_lengths = (
            "output/string_weighted_multi_categorical_features.lengths"
        )
        workspace.FeedBlob(output_feature_lengths, np.zeros(1, dtype=np.int32))
        output_feature_lengths_matrix = C2.ConstantFill(
            output_row_shape, value=1, dtype=caffe2_pb2.TensorProto.INT32
        )
        C2.net().FlattenToVec([output_feature_lengths_matrix], [output_feature_lengths])

        output_keys = "output/string_weighted_multi_categorical_features.values.keys"
        workspace.FeedBlob(output_keys, np.array(["a"]))
        C2.net().Tile([action_names, output_shape_row_count], [output_keys], axis=0)

        output_lengths_matrix = C2.ConstantFill(
            output_row_shape, value=len(actions), dtype=caffe2_pb2.TensorProto.INT32
        )
        output_lengths = (
            "output/string_weighted_multi_categorical_features.values.lengths"
        )
        workspace.FeedBlob(output_lengths, np.zeros(1, dtype=np.int32))
        C2.net().FlattenToVec([output_lengths_matrix], [output_lengths])

        output_values = (
            "output/string_weighted_multi_categorical_features.values.values"
        )
        workspace.FeedBlob(output_values, np.array([1.0]))
        C2.net().FlattenToVec([q_values], [output_values])
        return parameters, q_values
