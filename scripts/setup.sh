#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

pip uninstall -y horizon

# Generate thrift specs
thrift --gen py --out . ml/rl/thrift/core.thrift

# Install the current directory into python path
pip install -e .

# Run workflow tests
pytest ml/rl/test/workflow/test_oss_workflows.py
