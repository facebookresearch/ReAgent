#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Store model outputs here
mkdir outputs

python reagent/workflow/dqn_workflow.py -p reagent/workflow/sample_configs/discrete_action/dqn_example.json
