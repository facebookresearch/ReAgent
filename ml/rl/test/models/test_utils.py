#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import tempfile

import numpy.testing as npt
from caffe2.python import workspace
from caffe2.python.predictor.predictor_exporter import (
    prepare_prediction_net,
    save_to_db,
)


logger = logging.getLogger(__name__)


def check_save_load(
    self, model, expected_num_params, expected_num_inputs, expected_num_outputs
):
    pem, ws = model.get_predictor_export_meta_and_workspace()
    self.assertEqual(expected_num_params, len(pem.parameters))
    for p in pem.parameters:
        self.assertTrue(ws.HasBlob(p))
    self.assertEqual(expected_num_inputs, len(pem.inputs))
    self.assertEqual(expected_num_outputs, len(pem.outputs))

    input_prototype = model.input_prototype()

    with tempfile.TemporaryDirectory() as tmpdirname:
        db_path = os.path.join(tmpdirname, "model")
        logger.info("DB path: ", db_path)
        db_type = "minidb"
        with ws._ctx:
            save_to_db(db_type, db_path, pem)

        # Load the model from DB file and run it
        net = prepare_prediction_net(db_path, db_type)

        input_tensors = _flatten_named_tuple(input_prototype)
        input_names = model.input_blob_names()
        self.assertEqual(len(input_tensors), len(input_names))

        for name, tensor in zip(input_names, input_tensors):
            workspace.FeedBlob(name, tensor.numpy())

        workspace.RunNet(net)

        output_arrays = [workspace.FetchBlob(b) for b in model.output_blob_names()]
        output = model(input_prototype)
        output_tensors = _flatten_named_tuple(output)
        self.assertEqual(len(output_arrays), len(output_tensors))
        for a, t in zip(output_arrays, output_tensors):
            # FXIME: PyTorch and Caffe2 has slightly different operator implementation;
            # assert_array_equal would fail in some cases :(
            npt.assert_almost_equal(t.detach().numpy(), a)


def _flatten_named_tuple(nt):
    if not hasattr(nt, "_asdict"):
        # This is not a NamedTuple
        return [nt]
    nt = nt._asdict()
    return [e for v in nt.values() for e in _flatten_named_tuple(v) if e is not None]
