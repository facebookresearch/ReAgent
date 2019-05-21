#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import sys
import time

import numpy as np
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2, PytorchCaffe2Converter
from ml.rl.preprocessing import normalization
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from scipy import stats


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

NUM_FORWARD_PASSES = 50


def gen_data(
    num_binary_features=20,
    num_boxcox_features=20,
    num_continuous_features=300,
    num_enum_features=50,
    num_prob_features=50,
    num_quantile_features=20,
    num_continuous_action_features=0,
    samples_per_feature=10000,
):
    if num_quantile_features > 0:
        assert (
            samples_per_feature % 2 == 0
        ), "Samples per feature must be divisible by 2 if generating quantile features"
        quantile_samples = int(samples_per_feature / 2)

    np.random.seed(1)
    feature_value_map = {}
    feature_id = 1

    for _ in range(num_binary_features):
        feature_value_map[feature_id] = stats.bernoulli.rvs(
            0.5, size=samples_per_feature
        ).astype(np.float32)
        feature_id += 1

    for _ in range(num_boxcox_features):
        feature_value_map[feature_id] = stats.expon.rvs(
            size=samples_per_feature
        ).astype(np.float32)
        feature_id += 1

    for _ in range(num_continuous_features):
        feature_value_map[feature_id] = stats.norm.rvs(size=samples_per_feature).astype(
            np.float32
        )
        feature_id += 1

    for _ in range(num_enum_features):
        feature_value_map[feature_id] = (
            stats.randint.rvs(0, 10, size=samples_per_feature) * 1000
        ).astype(np.float32)
        feature_id += 1

    for _ in range(num_prob_features):
        feature_value_map[feature_id] = np.clip(
            stats.beta.rvs(a=2.0, b=2.0, size=samples_per_feature).astype(np.float32),
            0.01,
            0.99,
        )
        feature_id += 1

    for _ in range(num_quantile_features):
        feature_value_map[feature_id] = np.concatenate(
            (
                stats.norm.rvs(size=quantile_samples),
                stats.expon.rvs(size=quantile_samples),
            )
        ).astype(np.float32)
        feature_id += 1

    for _ in range(num_continuous_action_features):
        feature_value_map[feature_id] = stats.norm.rvs(size=samples_per_feature).astype(
            np.float32
        )
        feature_id += 1

    return feature_value_map


def benchmark(num_forward_passes):
    """
    Benchmark preprocessor speeds:
        1 - PyTorch
        2 - PyTorch -> ONNX -> C2
        3 - C2
    """

    feature_value_map = gen_data(
        num_binary_features=10,
        num_boxcox_features=10,
        num_continuous_features=10,
        num_enum_features=10,
        num_prob_features=10,
        num_quantile_features=10,
    )

    normalization_parameters = {}
    for name, values in feature_value_map.items():
        normalization_parameters[name] = normalization.identify_parameter(
            name, values, 10
        )

    sorted_features, _ = sort_features_by_normalization(normalization_parameters)

    # Dummy input
    input_matrix = np.zeros([10000, len(sorted_features)], dtype=np.float32)

    # PyTorch Preprocessor
    pytorch_preprocessor = Preprocessor(normalization_parameters, False)
    for i, feature in enumerate(sorted_features):
        input_matrix[:, i] = feature_value_map[feature]

    #################### time pytorch ############################
    start = time.time()
    for _ in range(NUM_FORWARD_PASSES):
        _ = pytorch_preprocessor.forward(input_matrix)
    end = time.time()
    logger.info(
        "PyTorch: {} forward passes done in {} seconds".format(
            NUM_FORWARD_PASSES, end - start
        )
    )

    ################ time pytorch -> ONNX -> caffe2 ####################
    buffer = PytorchCaffe2Converter.pytorch_net_to_buffer(
        pytorch_preprocessor, len(sorted_features), False
    )
    input_blob, output_blob, caffe2_netdef = PytorchCaffe2Converter.buffer_to_caffe2_netdef(
        buffer
    )
    torch_workspace = caffe2_netdef.workspace
    parameters = torch_workspace.Blobs()
    for blob_str in parameters:
        workspace.FeedBlob(blob_str, torch_workspace.FetchBlob(blob_str))
    torch_init_net = core.Net(caffe2_netdef.init_net)
    torch_predict_net = core.Net(caffe2_netdef.predict_net)
    input_matrix_blob = "input_matrix_blob"
    workspace.FeedBlob(input_blob, input_matrix)
    workspace.RunNetOnce(torch_init_net)
    start = time.time()
    for _ in range(NUM_FORWARD_PASSES):
        workspace.RunNetOnce(torch_predict_net)
        _ = workspace.FetchBlob(output_blob)
    end = time.time()
    logger.info(
        "PyTorch -> ONNX -> Caffe2: {} forward passes done in {} seconds".format(
            NUM_FORWARD_PASSES, end - start
        )
    )

    #################### time caffe2 ############################
    norm_net = core.Net("net")
    C2.set_net(norm_net)
    preprocessor = PreprocessorNet()
    input_matrix_blob = "input_matrix_blob"
    workspace.FeedBlob(input_matrix_blob, np.array([], dtype=np.float32))
    output_blob, _ = preprocessor.normalize_dense_matrix(
        input_matrix_blob, sorted_features, normalization_parameters, "", False
    )
    workspace.FeedBlob(input_matrix_blob, input_matrix)
    start = time.time()
    for _ in range(NUM_FORWARD_PASSES):
        workspace.RunNetOnce(norm_net)
        workspace.FetchBlob(output_blob)
    end = time.time()
    logger.info(
        "Caffe2: {} forward passes done in {} seconds".format(
            NUM_FORWARD_PASSES, end - start
        )
    )


if __name__ == "__main__":
    benchmark(NUM_FORWARD_PASSES)
