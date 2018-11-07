#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

set -e

. ${HOME}/miniconda/bin/activate
export LD_LIBRARY_PATH="${CONDA_PATH}/lib:${LD_LIBRARY_PATH}"

# Toggles between CUDA 8 or 9. Needs to be kept in sync with Dockerfile
TMP_CUDA_VERSION="9"

# Uninstall previous versions of PyTorch. Doing this twice is intentional.
# Error messages about torch not being installed are benign.
pip uninstall -y torch || true
pip uninstall -y torch || true
pip uninstall -y torch_nightly || true
pip uninstall -y torch_nightly || true
conda uninstall -y torch_nightly || true
conda uninstall -y torch_nightly || true

# anaconda doesn't have gym & onnx is lacking behind
pip install -r requirements.txt

conda install pytorch-nightly -c pytorch

# Install NCCL2.
wget "https://s3.amazonaws.com/pytorch/nccl_2.1.15-1%2Bcuda${TMP_CUDA_VERSION}.0_x86_64.txz"
TMP_NCCL_VERSION="nccl_2.1.15-1+cuda${TMP_CUDA_VERSION}.0_x86_64"
tar -xvf "${TMP_NCCL_VERSION}.txz"
export NCCL_ROOT_DIR="$(pwd)/${TMP_NCCL_VERSION}"
export LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
rm "${TMP_NCCL_VERSION}.txz"
