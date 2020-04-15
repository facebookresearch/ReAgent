#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import dataclasses
import typing

from reagent import parameters as rlp
from reagent.json_serialize import json_to_object, object_to_json
from reagent.test.base.horizon_test_base import HorizonTestBase


class TestJsonSerialize(HorizonTestBase):
    def test_json_serialize_basic(self):
        damp = rlp.DiscreteActionModelParameters(
            actions=["foo", "bar"],
            rl=rlp.RLParameters(),
            training=rlp.TrainingParameters(),
            rainbow=rlp.RainbowDQNParameters(double_q_learning=False, categorical=True),
            state_feature_params=None,
            target_action_distribution=[1.0, 2.0],
            evaluation=rlp.EvaluationParameters(),
        )
        self.assertEqual(
            damp,
            json_to_object(object_to_json(damp), rlp.DiscreteActionModelParameters),
        )

    def test_json_serialize_nested(self):
        @dataclasses.dataclass
        class Test1:
            x: int

        @dataclasses.dataclass
        class Test2:
            x: typing.List[Test1]
            y: typing.Dict[str, Test1]

        t = Test2(x=[Test1(x=3), Test1(x=4)], y={"1": Test1(x=5), "2": Test1(x=6)})
        self.assertEqual(t, json_to_object(object_to_json(t), Test2))
