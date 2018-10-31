#!/usr/bin/env bash

# Generate thrift specs
thrift --gen py --out . ml/rl/thrift/core.thrift

# Install the current directory into python path
pip install -e .

# Run workflow tests
python -m ml.rl.test.workflow.test_oss_workflows
