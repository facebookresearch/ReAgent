#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import random
import unittest
from typing import Callable

import numpy as np
import torch
from reagent.core.configuration import make_config_class
from reagent.tensorboardX import SummaryWriterContext
from ruamel.yaml import YAML


SEED = 0


class HorizonTestBase(unittest.TestCase):
    def setUp(self):
        SummaryWriterContext._reset_globals()
        logging.basicConfig(level=logging.INFO)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    def tearDown(self):
        SummaryWriterContext._reset_globals()

    @classmethod
    def run_from_config(cls, run_test: Callable, config_path: str, use_gpu: bool):
        yaml = YAML(typ="safe")
        with open(config_path, "r") as f:
            config_dict = yaml.load(f.read())
        config_dict["use_gpu"] = use_gpu

        @make_config_class(run_test)
        class ConfigClass:
            pass

        config = ConfigClass(**config_dict)  # type: ignore
        return run_test(**config.asdict())  # type: ignore
