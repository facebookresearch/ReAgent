#!/usr/bin/env bash

# Store model outputs here
mkdir outputs

python ml/rl/workflow/dqn_workflow.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json
