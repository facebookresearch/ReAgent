#!/usr/bin/env bash

# Install the current directory into python path
pip install -e .

# Run workflow tests
python -m unittest -b ml.rl.test.workflow.test_oss_workflows
