#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds ReAgent and runs basic tests.

set -ex

export PATH=${HOME}/miniconda/bin:$PATH

pip uninstall -y reagent

# Installing from current directory, any update will be reflected system-wide
pip install -e .

# Clone submodules
git submodule update --force --recursive --init --remote

# Build RASP
mkdir -p serving/build
pushd serving/build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
make -j2
popd

# Run RASP tests
serving/build/RaspTest
