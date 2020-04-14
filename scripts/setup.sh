#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

set -ex

pip uninstall -y reagent

# Install the current directory into python path
pip install -e .

# Build RASP and run tests
mkdir -p serving/build
pushd serving/build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
make -j4
make test
popd

# Run workflow tests
pytest reagent/test/workflow/test_oss_workflows.py
