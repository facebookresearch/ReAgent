#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

import numpy as np

from caffe2.python import core, workspace

from ml.rl.thrift.eval.ttypes import (
    PolicyEvaluatorParameters,
)

from ml.rl.caffe_utils import C2
from ml.rl.evaluation.policy_simulator import PolicySimulator


class PolicyEvaluator(object):
    def __init__(
        self,
        params: PolicyEvaluatorParameters,
        db_type: str,
    ) -> None:
        self.params = params
        self.process_slate_net = core.Net('policy_evaluator')
        C2.set_net(self.process_slate_net)
        self.action_probabilities = PolicySimulator.plan(
            self.process_slate_net,
            params,
            db_type,
        )
        self.created_net = False

    def evaluate_slates(
        self,
        model_input_set: List[List[Dict[str, np.ndarray]]],
        action_selection: np.ndarray,
        rewards: np.ndarray,
        baseline_probabilities: np.ndarray,
    ) -> float:
        num_examples = len(model_input_set)
        value_sum = 0.0
        for slate_index, model_inputs in enumerate(model_input_set):
            for model_index, model_input in enumerate(model_inputs):
                value_model_params = self.params.value_models[model_index]
                for name, value in model_input.items():
                    print(
                        "FEEDING",
                        '{}_{}'.format(value_model_params.name, name)
                    )
                    workspace.FeedBlob(
                        '{}_{}'.format(value_model_params.name, name), value
                    )
            if not self.created_net:
                workspace.CreateNet(self.process_slate_net)
                self.created_net = True
            workspace.RunNet(self.process_slate_net)
            action_probabilities = workspace.FetchBlob(
                self.action_probabilities
            )
            ips_numerator = rewards[slate_index] * \
                action_probabilities[action_selection[slate_index]]
            ips_denominator = baseline_probabilities[slate_index]
            value_sum += ips_numerator / ips_denominator
            logger.debug(
                'DETAILS',
                rewards[slate_index],
                action_probabilities[action_selection[slate_index]],
                ips_denominator,
                value_sum,
            )
        return value_sum / num_examples
