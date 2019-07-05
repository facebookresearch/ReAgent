#!/bin/bash

set -ex

# This has to be kept in sync with the docker file
CUDA_VERSION=9.2
NCCL_UBUNTU_VER=ubuntu1804
NCCL_DEB='nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb'

# The deb is agnostic of CUDA version
curl -LO "http://developer.download.nvidia.com/compute/machine-learning/repos/${NCCL_UBUNTU_VER}/x86_64/${NCCL_DEB}"

# This dpkg call needs wget
apt-get update
apt-get install -y wget
dpkg -i "${NCCL_DEB}"

NCCL_LIB_VERSION="2.3.4-1+cuda${CUDA_VERSION:0:3}"

apt update
apt install -y --allow-downgrades --allow-change-held-packages libnccl2=$NCCL_LIB_VERSION libnccl-dev=$NCCL_LIB_VERSION
