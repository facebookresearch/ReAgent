#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds ReAgent and runs basic tests.

set -ex

export PATH=${HOME}/miniconda/bin:$PATH

pip uninstall -y reagent

# Installing from current directory, any update will be reflected system-wide
pip install -e .
pytest
