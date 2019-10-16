#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
from typing import NamedTuple

import numpy as np
from caffe2.python import core


class OutputTransformerNet(NamedTuple):
    net: core.Net
    init_net: core.Net


class OutputTransformerBase(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_net(self, original_output) -> OutputTransformerNet:
        pass

    def create_const(
        self, init_net, name, value, shape=None, dtype=core.DataType.FLOAT
    ):
        shape = shape or []
        blob = init_net.NextScopedBlob(name)
        if not isinstance(value, list):
            value = [value]
        init_net.GivenTensorFill([], blob, shape=shape, values=value, dtype=dtype)
        init_net.AddExternalOutput(blob)
        return blob

    def get_batch_size_blob(self, net, data):
        data_shape = net.Shape(data, net.NextScopedBlob("data_shape"))
        batch_size = net.Slice(
            data_shape, net.NextScopedBlob("batch_size"), starts=[0], ends=[1]
        )
        return batch_size

    def create_action_name_blob(self, init_net, action_names):
        return self.create_const(
            init_net,
            "action_names",
            action_names,
            shape=[len(action_names)],
            dtype=core.DataType.STRING,
        )

    def export_q_values(self, net, q_values, action_names, action_name_blob):
        batch_size = self.get_batch_size_blob(net, q_values)
        feature_lengths_blob = core.BlobReference(
            "output/string_weighted_multi_categorical_features.lengths"
        )
        net.ConstantFill(
            batch_size,
            feature_lengths_blob,
            value=1,
            dtype=core.DataType.INT32,
            input_as_shape=1,
        )
        feature_keys_blob = core.BlobReference(
            "output/string_weighted_multi_categorical_features.keys"
        )
        net.ConstantFill(
            batch_size,
            feature_keys_blob,
            value=0,
            dtype=core.DataType.INT64,
            input_as_shape=1,
        )
        values_lengths_blob = core.BlobReference(
            "output/string_weighted_multi_categorical_features.values.lengths"
        )
        net.ConstantFill(
            batch_size,
            values_lengths_blob,
            value=len(action_names),
            dtype=core.DataType.INT32,
            input_as_shape=1,
        )
        values_keys_blob = core.BlobReference(
            "output/string_weighted_multi_categorical_features.values.keys"
        )
        net.Tile([action_name_blob, batch_size], values_keys_blob, axis=0)
        values_values_blob = core.BlobReference(
            "output/string_weighted_multi_categorical_features.values.values"
        )
        net.FlattenToVec(q_values, values_values_blob)
        net.AddExternalOutput(
            feature_lengths_blob,
            feature_keys_blob,
            values_lengths_blob,
            values_keys_blob,
            values_values_blob,
        )


class ActorOutputTransformer(OutputTransformerBase):
    def __init__(
        self,
        action_feature_ids,
        serving_max_scale,
        serving_min_scale,
        training_max_scale=None,
        training_min_scale=None,
    ):
        self.action_feature_ids = action_feature_ids
        self.serving_max_scale = np.array(serving_max_scale, dtype=np.float)
        self.serving_min_scale = np.array(serving_min_scale, dtype=np.float)
        self.training_max_scale = np.array(
            training_max_scale or [1.0 - 1e-6] * len(action_feature_ids), dtype=np.float
        )
        self.training_min_scale = np.array(
            training_min_scale or [-1.0 + 1e-6] * len(action_feature_ids),
            dtype=np.float,
        )

    def create_net(self, original_output):
        net = core.Net("output_transformer")
        init_net = core.Net("output_transformer_init")

        action = original_output.action()

        batch_size = self.get_batch_size_blob(net, action)

        action_dims = self.create_const(
            init_net,
            "action_dims",
            len(self.action_feature_ids),
            shape=[1],
            dtype=core.DataType.INT32,
        )
        lengths = core.BlobReference("output/float_features.lengths")
        net.Tile([action_dims, batch_size], lengths, axis=0)
        net.AddExternalOutput(lengths)

        action_feature_ids = self.create_const(
            init_net,
            "action_feature_ids",
            self.action_feature_ids,
            shape=[len(self.action_feature_ids)],
            dtype=core.DataType.INT64,
        )
        keys = core.BlobReference("output/float_features.keys")
        net.Tile([action_feature_ids, batch_size], keys, axis=0)
        net.AddExternalOutput(keys)

        values = core.BlobReference("output/float_features.values")

        # Shifting action to [training_max - training_min, 0]
        training_min_scale = self.create_const(
            init_net,
            "training_min_scale",
            self.training_min_scale.tolist(),
            shape=[len(self.training_min_scale)],
        )
        shifted_action = net.Sub([action, training_min_scale], 1, broadcast=1)
        # Scaling action by (serving_max - serving_min) / (trainig_max - trainig_min)
        scaling_factor = (self.serving_max_scale - self.serving_min_scale) / (
            self.training_max_scale - self.training_min_scale
        )
        scaling_factor_blob = self.create_const(
            init_net,
            "scaling_factor",
            scaling_factor.tolist(),
            shape=[len(scaling_factor)],
        )
        scaled_shifted_action = net.Mul(
            [shifted_action, scaling_factor_blob], 1, broadcast=1
        )
        # Shifting action to [serving_max, serving_min]
        serving_min_scale = self.create_const(
            init_net,
            "serving_min_scale",
            self.serving_min_scale.tolist(),
            shape=[len(self.serving_min_scale)],
        )
        scaled_action = net.Add(
            [scaled_shifted_action, serving_min_scale], 1, broadcast=1
        )

        # Now, we can flatten and return
        net.FlattenToVec(scaled_action, values)
        net.AddExternalOutput(values)

        return OutputTransformerNet(net=net, init_net=init_net)
