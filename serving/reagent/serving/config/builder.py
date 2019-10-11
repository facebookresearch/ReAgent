#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from typing import Any, Dict

import reagent.serving.config.config as config
import reagent.serving.config.namespace as namespace
import reagent.serving.config.operators as defaults
from reagent.serving.config.serialize import config_to_json


DECISION_PLANS = {}


def export(app_id: str, configs: Dict[str, Dict[str, Any]]):
    DECISION_PLANS[app_id] = configs


class DecisionPlanBuilder(object):
    def __init__(self):
        self.operators = []
        self.constants = []
        self.root_op = None
        self.num_actions_to_choose = 1
        self.reward_function = ""
        self.reward_aggregator = config.DecisionRewardAggreation.DRA_MAX

    def set_root(self, op):
        self.root_op = op
        return self

    def set_num_actions_to_choose(self, num):
        self.num_actions_to_choose = num
        return self

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function
        return self

    def set_reward_aggregator(self, reward_aggregator):
        self.reward_aggregator = reward_aggregator
        return self

    def serialize(self):
        visited = set()
        next_id = [0]

        def _id():
            next_id[0] += 1
            return next_id[0]

        def create_node(node):
            constant = None
            op = None

            if isinstance(node, namespace.DecisionOperator):
                if node in visited:
                    return node.name
                else:
                    node.name = "{}_{}".format(node.op_name, _id())
                    visited.add(node)
                input_dep_map = {}
                for name, arg in node.arguments().items():
                    if arg is not None:
                        input_dep_map[name] = create_node(arg)
                op = config.Operator(
                    name=node.name, op_name=node.op_name, input_dep_map=input_dep_map
                )
            else:
                constant = config.Constant(name="constant_{}".format(_id()), value=node)

            if constant is not None:
                self.constants.append(constant)
                return constant.name
            elif op is not None:
                self.operators.append(op)
                return op.name
            else:
                raise TypeError("Invalid type: {}".format(node))

        if self.root_op is None:
            raise ValueError("Need to set root operator before exporting")

        if not isinstance(self.root_op, namespace.DecisionOperator):
            raise ValueError("The root node has to be an operator")

        create_node(self.root_op)

    def build(self):
        self.serialize()
        decision_config = config.DecisionConfig(
            operators=self.operators,
            constants=self.constants,
            num_actions_to_choose=self.num_actions_to_choose,
            reward_function=self.reward_function,
            reward_aggregator=self.reward_aggregator,
        )

        return config_to_json(config.DecisionConfig, decision_config)


globals().update(vars(defaults))
