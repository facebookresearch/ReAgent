#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
import six
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
from caffe2.python.predictor.predictor_exporter import (
    load_from_db,
    prepare_prediction_net,
    save_to_db,
)
from caffe2.python.predictor.predictor_py_utils import GetBlobs
from caffe2.python.predictor_constants import predictor_constants
from ml.rl.caffe_utils import C2


logger = logging.getLogger(__name__)


class RLPredictor:
    def __init__(self, net, parameters, int_features=False):
        """

        :param net caffe2 net used for prediction
        :param input_blobs caffe2 blobs used as input
        :param output_blobs caffe2 blobs used as output
        :param parameters caffe2 blobs used as network paramers
        :param int_features boolean indicating if int_features blob will be present
        """
        self._net = net
        self._input_blobs = [
            "input/float_features.lengths",
            "input/float_features.keys",
            "input/float_features.values",
        ]
        if int_features:
            self._input_blobs.extend(
                [
                    "input/int_features.lengths",
                    "input/int_features.keys",
                    "input/int_features.values",
                ]
            )
        self._output_blobs = [
            "output/string_weighted_multi_categorical_features.keys",
            "output/string_weighted_multi_categorical_features.lengths",
            "output/string_weighted_multi_categorical_features.values.keys",
            "output/string_weighted_multi_categorical_features.values.lengths",
            "output/string_weighted_multi_categorical_features.values.values",
        ]
        self._parameters = parameters
        self.is_discrete = None

    def policy(self, float_state_features, int_state_features=None) -> np.ndarray:
        """ Returns np array of action names to take for each state
        :param float_state_features A list of feature -> float value dict examples
        :param int_state_features A list of feature -> int value dict examples
        """
        float_state_keys = []
        float_state_values = []
        for example in float_state_features:
            for k, v in example.items():
                float_state_keys.append(k)
                float_state_values.append(v)
        workspace.FeedBlob(
            "input/float_features.lengths",
            np.array([len(e) for e in float_state_features], dtype=np.int32),
        )
        workspace.FeedBlob(
            "input/float_features.keys",
            np.array(float_state_keys, dtype=np.int32).flatten(),
        )
        workspace.FeedBlob(
            "input/float_features.values",
            np.array(float_state_values, dtype=np.float32).flatten(),
        )

        if int_state_features is not None:
            int_state_keys = []
            int_state_values = []
            for example in int_state_features:
                for k, v in example.items():
                    int_state_keys.append(k)
                    int_state_values.append(v)
            workspace.FeedBlob(
                "input/int_features.lengths",
                np.array([len(e) for e in int_state_features], dtype=np.int32),
            )
            workspace.FeedBlob(
                "input/int_features.keys",
                np.array(int_state_keys, dtype=np.int64).flatten(),
            )
            workspace.FeedBlob(
                "input/int_features.values",
                np.array(int_state_values, dtype=np.int32).flatten(),
            )

        workspace.RunNet(self._net)

        if self.is_discrete:
            # discrete action policy has string values (action names)
            return workspace.FetchBlob(
                "output/string_single_categorical_features.values"
            )
        elif not self.is_discrete:
            # [a1_maxq, a1_softmax, a2_maxq, a2_softmax, ...]
            # parametric action policy has int values (action indexes)
            return workspace.FetchBlob("output/int_single_categorical_features.values")
        else:
            raise Exception(
                "is_discrete property {} not valid".format(self.is_discrete)
            )

    def predict(self, float_state_features, int_state_features=None):
        """ Returns values for each state
        :param float_state_features A list of feature -> float value dict examples
        :param int_state_features A list of feature -> int value dict examples
        """
        float_state_keys = []
        float_state_values = []
        for example in float_state_features:
            for k, v in example.items():
                float_state_keys.append(k)
                float_state_values.append(v)
        workspace.FeedBlob(
            "input/float_features.lengths",
            np.array([len(e) for e in float_state_features], dtype=np.int32),
        )
        workspace.FeedBlob(
            "input/float_features.keys", np.array(float_state_keys, dtype=np.int64)
        )
        workspace.FeedBlob(
            "input/float_features.values",
            np.array(float_state_values, dtype=np.float32),
        )

        if int_state_features is not None:
            workspace.FeedBlob(
                "input/int_features.lengths",
                np.array([len(e) for e in int_state_features], dtype=np.int32),
            )
            int_state_keys = []
            int_state_values = []
            for example in int_state_features:
                for k, v in example.items():
                    int_state_keys.append(k)
                    int_state_values.append(v)
            workspace.FeedBlob(
                "input/int_features.keys",
                np.array(int_state_keys, dtype=np.int64).flatten(),
            )
            workspace.FeedBlob(
                "input/int_features.values",
                np.array(int_state_values, dtype=np.int32).flatten(),
            )

        workspace.RunNet(self._net)

        output_lengths = workspace.FetchBlob(
            "output/string_weighted_multi_categorical_features.values.lengths"
        )
        output_names = workspace.FetchBlob(
            "output/string_weighted_multi_categorical_features.values.keys"
        )
        output_values = workspace.FetchBlob(
            "output/string_weighted_multi_categorical_features.values.values"
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
        pass

    def save(self, db_path, db_type):
        """ Saves network to db

        :param db_path see save_to_db
        :param db_type see save_to_db
        """
        meta = self.get_predictor_export_meta()
        for parameter in self._parameters:
            parameter_data = workspace.FetchBlob(parameter)
            logger.info("DATA TYPE " + parameter_data.dtype.kind)
            if parameter_data.dtype.kind in {"U", "S", "O"}:
                continue  # Don't bother checking string blobs for nan
            logger.info("Checking parameter {} for nan".format(parameter))
            if np.any(np.isnan(parameter_data)):
                logger.info("WARNING: parameter {} is nan".format(parameter))
        save_to_db(db_type, db_path, meta)

    @classmethod
    def load(cls, db_path, db_type, int_features=False):
        """ Creates Predictor by loading from a database

        :param db_path see load_from_db
        :param db_type see load_from_db
        :param int_features bool indicating if int_features are present
        """
        net = prepare_prediction_net(db_path, db_type)
        meta = load_from_db(db_path, db_type)
        parameters = GetBlobs(meta, predictor_constants.PARAMETERS_BLOB_TYPE)
        return cls(net, parameters, int_features)

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
    def _forward_pass(cls, model, trainer, normalized_dense_matrix, actions):
        C2.set_model(model)

        parameters = []
        q_values = "q_values"
        workspace.FeedBlob(q_values, np.zeros(1, dtype=np.float32))
        trainer.build_predictor(model, normalized_dense_matrix, q_values)
        parameters.extend(model.GetAllParams())

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
        C2.net().Tile([action_names, output_shape_row_count], [output_keys], axis=1)

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
