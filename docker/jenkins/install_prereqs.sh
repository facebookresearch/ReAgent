#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Assuming that passing an argument means to use GPU
gpu="$1"

set -e

export LD_LIBRARY_PATH="${CONDA_PATH}/lib:${LD_LIBRARY_PATH}"

# Toggles between CUDA 8 or 9. Needs to be kept in sync with Dockerfile
TMP_CUDA_VERSION="9"

if [ -z "$gpu" ]; then
    conda install pytorch-nightly-cpu -c pytorch
else
    conda install pytorch-nightly -c pytorch
fi

# Install NCCL2.
wget "https://s3.amazonaws.com/pytorch/nccl_2.1.15-1%2Bcuda${TMP_CUDA_VERSION}.0_x86_64.txz"
TMP_NCCL_VERSION="nccl_2.1.15-1+cuda${TMP_CUDA_VERSION}.0_x86_64"
tar -xvf "${TMP_NCCL_VERSION}.txz"
export NCCL_ROOT_DIR="$(pwd)/${TMP_NCCL_VERSION}"
export LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
rm "${TMP_NCCL_VERSION}.txz"

