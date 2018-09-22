#!/bin/bash

set -e

. ${HOME}/miniconda/bin/activate
export LD_LIBRARY_PATH="${CONDA_PATH}/lib:${LD_LIBRARY_PATH}"

# Toggles between CUDA 8 or 9. Needs to be kept in sync with Dockerfile
TMP_CUDA_VERSION="9"

# Uninstall previous versions of PyTorch. Doing this twice is intentional.
# Error messages about torch not being installed are benign.
pip uninstall -y torch || true
pip uninstall -y torch || true

pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install -r requirements.txt

# # Install basic PyTorch dependencies.
# # Numpy should be > 1.12 to prevent torch tensor from treating single-element value as shape.
# conda install -y cffi cmake mkl mkl-include numpy=1.15.1 pyyaml setuptools typing tqdm
# # Add LAPACK support for the GPU.
# conda install -y -c pytorch "magma-cuda${TMP_CUDA_VERSION}0"

# # Caffe2 relies on the past module.
# yes | pip install future

# # Install NCCL2.
# wget "https://s3.amazonaws.com/pytorch/nccl_2.1.15-1%2Bcuda${TMP_CUDA_VERSION}.0_x86_64.txz"
# TMP_NCCL_VERSION="nccl_2.1.15-1+cuda${TMP_CUDA_VERSION}.0_x86_64"
# tar -xvf "${TMP_NCCL_VERSION}.txz"
# export NCCL_ROOT_DIR="$(pwd)/${TMP_NCCL_VERSION}"
# export LD_LIBRARY_PATH="${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
# rm "${TMP_NCCL_VERSION}.txz"


# # Use the combined PyTorch/Caffe2 package instead of rebuilding from source.
# conda install -y -c caffe2 "pytorch-caffe2-cuda${TMP_CUDA_VERSION}.0-cudnn7"
# # Force re-install of numpy 1.15.1
# conda install -y numpy==1.15.1 --no-deps --force

# # echo "Starting to install ONNX"
# # git clone --recursive https://github.com/onnx/onnx.git
# # yes | pip install ./onnx 2>&1 | tee ONNX_OUT

# # train with tensorboard
# # yes | pip install tensorboard_logger

# yes | pip install -r requirements.txt
