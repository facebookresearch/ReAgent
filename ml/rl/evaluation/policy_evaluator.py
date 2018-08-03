#!/usr/bin/env python3

import logging
from typing import Dict, List

import numpy as np
from caffe2.python import core, workspace
from caffe2.python.predictor.predictor_exporter import PredictorExportMeta, save_to_db
from ml.rl.caffe_utils import C2
from ml.rl.evaluation.policy_simulator import PolicySimulator
from ml.rl.thrift.eval.ttypes import (
    PolicyEvaluatorParameters,
    ValueInputModelParameters,
)


logger = logging.getLogger(__name__)


def save_sum_deterministic_policy(model_names, path, db_type):
    net = core.Net("DeterministicPolicy")
    C2.set_net(net)
    output = "ActionProbabilities"
    workspace.FeedBlob(output, np.array([1.0]))
    model_outputs = []
    for model in model_names:
        model_output = "{}_Output".format(model)
        workspace.FeedBlob(model_output, np.array([[1.0]], dtype=np.float32))
        model_outputs.append(model_output)
    max_action = C2.FlattenToVec(C2.ArgMax(C2.Transpose(C2.Sum(*model_outputs))))
    one_blob = C2.NextBlob("one")
    workspace.FeedBlob(one_blob, np.array([1.0], dtype=np.float32))
    C2.net().SparseToDense([max_action, one_blob, model_outputs[0]], [output])
    meta = PredictorExportMeta(net, [one_blob], model_outputs, [output])
    save_to_db(db_type, path, meta)


class Slate(object):
    __slot__ = ["policy_net_features", "model_feature_set"]

    def __init__(
        self,
        policy_net_features: Dict[str, np.ndarray],
        model_feature_set: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        self.policy_net_features = policy_net_features
        self.model_feature_set = model_feature_set


class PolicyEvaluator(object):
    def __init__(self, params: PolicyEvaluatorParameters) -> None:
        self.params = params
        self.process_slate_net = core.Net("policy_evaluator")
        C2.set_net(self.process_slate_net)
        self.action_probabilities = PolicySimulator.plan(
            self.process_slate_net, params, self.params.db_type
        )
        self.created_net = False
        self.value_input_models: Dict[str, ValueInputModelParameters] = {}
        for model in self.params.value_input_models:
            self.value_input_models[model.name] = model

    def evaluate_slates(
        self,
        slates: List[Slate],
        action_selection: np.ndarray,
        rewards: np.ndarray,
        baseline_probabilities: np.ndarray,
    ) -> float:
        num_slates = len(slates)
        value_sum = 0.0
        for slate_index, slate in enumerate(slates):
            for name, value in slate.policy_net_features.items():
                workspace.FeedBlob(name, value)
            for model_name, model_input in slate.model_feature_set.items():
                value_input_model_params = self.value_input_models[model_name]
                print(model_input)
                for name, value in model_input.items():
                    workspace.FeedBlob(
                        "{}_{}".format(value_input_model_params.name, name), value
                    )
            if not self.created_net:
                workspace.CreateNet(self.process_slate_net)
                self.created_net = True
            workspace.RunNet(self.process_slate_net)
            action_probabilities = workspace.FetchBlob(self.action_probabilities)
            ips_numerator = (
                rewards[slate_index]
                * action_probabilities[action_selection[slate_index]]
            )
            ips_denominator = baseline_probabilities[slate_index]
            value_sum += ips_numerator / ips_denominator
        return value_sum / num_slates
