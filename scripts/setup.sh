#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

set -ex

pip uninstall -y horizon

# Install the current directory into python path
pip install -e .

# Build RASP and run tests
mkdir -p serving/build
pushd serving/build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
make -j`nproc`
make test
popd

# Run workflow tests
pytest ml/rl/test/workflow/test_oss_workflows.py
