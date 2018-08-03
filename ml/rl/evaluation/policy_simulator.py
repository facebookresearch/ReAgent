#!/usr/bin/env python3

import logging

from caffe2.python import core, workspace
from caffe2.python.predictor.predictor_exporter import (
    load_from_db,
    prepare_prediction_net,
)
from caffe2.python.predictor.predictor_py_utils import GetBlobs
from caffe2.python.predictor_constants import predictor_constants
from ml.rl.thrift.eval.ttypes import PolicyEvaluatorParameters


logger = logging.getLogger(__name__)


class PolicySimulator(object):
    @classmethod
    def plan(
        cls, net, simulator_parameters: PolicyEvaluatorParameters, db_type: str
    ) -> str:
        for name, value in simulator_parameters.global_value_inputs:
            workspace.FeedBlob(name, value)

        for model in simulator_parameters.value_input_models:
            if model.path is not None and len(model.path) > 0:
                model_net = prepare_prediction_net(model.path, db_type)

                # By default, inputs are not remapped, so let's force a remapping
                meta = load_from_db(model.path, db_type)
                model_net_inputs = GetBlobs(meta, predictor_constants.INPUTS_BLOB_TYPE)
                remap = {}
                for inp in model_net_inputs:
                    remap[inp] = "{}_{}".format(model.name, inp)
                logger.debug("REMAP VALUE MODEL: {}".format(remap))
                model_net, _ = core.clone_and_bind_net(
                    model_net,
                    "new_{}".format(model.name),
                    "{}_".format(model.name),
                    blob_remap=remap,
                )
                net.AppendNet(model_net)

        policy_net = prepare_prediction_net(
            simulator_parameters.propensity_net_path, db_type
        )
        meta = load_from_db(simulator_parameters.propensity_net_path, db_type)
        policy_net_outputs = GetBlobs(meta, predictor_constants.OUTPUTS_BLOB_TYPE)
        assert policy_net_outputs == ["ActionProbabilities"]
        net.AppendNet(policy_net)
        return "ActionProbabilities"
