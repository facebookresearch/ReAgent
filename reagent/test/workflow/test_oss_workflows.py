#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import os
import unittest
import zipfile
from typing import Dict
from unittest.mock import patch

import reagent
import reagent.workflow.cli as cli
import torch
from click.testing import CliRunner
from reagent.parameters import NormalizationParameters
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.workflow.types import Dataset
from ruamel.yaml import YAML


base_dir = os.path.abspath(os.path.dirname(reagent.__file__))
curr_dir = os.path.abspath(os.path.dirname(__file__))

CARTPOLE_NORMALIZATION_JSON = os.path.join(
    curr_dir, "test_data/discrete_action/cartpole_norm.json"
)
DQN_WORKFLOW_PARQUET_ZIP = os.path.join(
    curr_dir, "test_data/discrete_action/dqn_workflow.zip"
)
DQN_WORKFLOW_PARQUET_REL_PATH = "dqn_workflow"
DQN_WORKFLOW_YAML = os.path.join(
    base_dir, "workflow/sample_configs/discrete_dqn_cartpole_offline.yaml"
)

# where to store config for testing cli
NEW_CONFIG_NAME = "config.yaml"

# module to patch
DISCRETE_DQN_BASE = "reagent.workflow.model_managers.discrete_dqn_base"


def get_test_workflow_config(path_to_config: str, use_gpu: bool):
    """ Loads and modifies config to fun fast. """
    yaml = YAML(typ="safe")
    with open(path_to_config, "r") as f:
        config = yaml.load(f)
        config["use_gpu"] = use_gpu
        config["num_train_epochs"] = 1
        config["num_eval_episodes"] = 1
        # minimum score is 0
        config["passing_score_bar"] = -0.0001
        # both table and eval_table will be referenced to our mocked parquet
        config["input_table_spec"]["table_sample"] = 50.0
        config["input_table_spec"]["eval_table_sample"] = 50.0
    return config


def mock_cartpole_normalization() -> Dict[int, NormalizationParameters]:
    """ Get mock normalization from our local file. """
    with open(CARTPOLE_NORMALIZATION_JSON, "r") as f:
        norm = json.load(f)

    norm_params_dict = {}
    for k, v in norm.items():
        norm_params_dict[k] = NormalizationParameters(**json.loads(v))
    return norm_params_dict


class TestOSSWorkflows(HorizonTestBase):
    """ Run workflow to ensure no crashes, correctness/performance not tested. """

    def _test_dqn_workflow(self, use_gpu=False, use_all_avail_gpus=False):
        runner = CliRunner()
        config = get_test_workflow_config(
            path_to_config=DQN_WORKFLOW_YAML, use_gpu=use_gpu
        )

        # create new altered config (for faster testing)
        with runner.isolated_filesystem():
            yaml = YAML(typ="safe")
            with open(NEW_CONFIG_NAME, "w") as f:
                yaml.dump(config, f)

            # unzip zipped parquet folder into cwd
            with zipfile.ZipFile(DQN_WORKFLOW_PARQUET_ZIP, "r") as zip_ref:
                zip_ref.extractall()

            # patch the two calls to spark
            # dataset points to the unzipped parquet folder
            # normalization points to mocked norm extracted from json
            mock_dataset = Dataset(
                parquet_url=f"file://{os.path.abspath(DQN_WORKFLOW_PARQUET_REL_PATH)}"
            )
            mock_normalization = mock_cartpole_normalization()
            with patch(
                f"{DISCRETE_DQN_BASE}.query_data", return_value=mock_dataset
            ), patch(
                f"{DISCRETE_DQN_BASE}.identify_normalization_parameters",
                return_value=mock_normalization,
            ):
                # call the cli test
                result = runner.invoke(
                    cli.run,
                    [
                        "reagent.workflow.gym_batch_rl.train_and_evaluate_gym",
                        NEW_CONFIG_NAME,
                    ],
                    catch_exceptions=False,
                )

            print(result.output)
            assert result.exit_code == 0, f"result = {result}"

    def test_dqn_workflow(self):
        self._test_dqn_workflow()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_dqn_workflow_gpu(self):
        self._test_dqn_workflow(use_gpu=True)


if __name__ == "__main__":
    unittest.main()
