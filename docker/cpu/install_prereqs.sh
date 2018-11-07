#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

set -e

. ${HOME}/miniconda/bin/activate
export LD_LIBRARY_PATH="${CONDA_PATH}/lib:${LD_LIBRARY_PATH}"

# Uninstall previous versions of PyTorch. Doing this twice is intentional.
# Error messages about torch not being installed are benign.
pip uninstall -y torch || true
pip uninstall -y torch || true
pip uninstall -y torch_nightly || true
pip uninstall -y torch_nightly || true
conda uninstall -y torch_nightly || true
conda uninstall -y torch_nightly || true

pip install -r requirements.txt

conda install pytorch-nightly-cpu -c pytorch
