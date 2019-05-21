# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Pre-req installations:
# https://docs.docker.com/install/linux/docker-ce/ubuntu/
# https://github.com/NVIDIA/nvidia-docker

# Usage:
#        sudo docker build -t horizon_initial_release . 2>&1 | tee stdout
#    or
#        sudo nvidia-docker build -t horizon_initial_release . 2>&1 | tee stdout
#        sudo nvidia-docker run -i -t --rm horizon_initial_release /bin/bash

# Remove all stopped Docker containers:   sudo docker rm $(sudo docker ps -a -q)
# Remove all untagged images:             sudo docker rmi $(sudo docker images -q --filter "dangling=true")

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  git \
  sudo \
  software-properties-common \
  vim \
  emacs \
  wget

# Sometimes needed to avoid SSL CA issues.
RUN update-ca-certificates

ENV HOME /home
WORKDIR ${HOME}/

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
  chmod +x miniconda.sh && \
  ./miniconda.sh -b -p ${HOME}/miniconda && \
  rm miniconda.sh

# Setting these env var outside of the install script to ensure
# they persist in image
# (See https://stackoverflow.com/questions/33379393/docker-env-vs-run-export)
ENV PATH ${HOME}/miniconda/bin:$PATH
ENV CONDA_PATH ${HOME}/miniconda
ENV LD_LIBRARY_PATH ${CONDA_PATH}/lib:${LD_LIBRARY_PATH}

# Set channels
RUN conda config --add channels conda-forge # For ONNX/tensorboardX
RUN conda config --add channels pytorch # For PyTorch

# Add files to image
ADD requirements.txt requirements.txt
ADD preprocessing/pom.xml /tmp/pom.xml

# Install dependencies
RUN conda install --file requirements.txt
RUN rm requirements.txt

# Install open ai gym
RUN pip install "gym[classic_control,box2d,atari]"

RUN conda install cudatoolkit=9.0 -c pytorch

# Set JAVA_HOME for Spark
ENV JAVA_HOME ${HOME}/miniconda

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-2.3.3/spark-2.3.3-bin-hadoop2.7.tgz && \
  tar -xzf spark-2.3.3-bin-hadoop2.7.tgz && \
  mv spark-2.3.3-bin-hadoop2.7 /usr/local/spark

# Caches dependencies so they do not need to be re-downloaded
RUN mvn -f /tmp/pom.xml dependency:resolve

# Clean up pom.xml
RUN rm /tmp/pom.xml

# Reminder: this should be updated when switching between CUDA 8 or 9. Should
# be kept in sync with TMP_CUDA_VERSION in install_prereqs.sh
ENV NCCL_ROOT_DIR ${HOME}/horizon/nccl_2.1.15-1+cuda9.0_x86_64
ENV LD_LIBRARY_PATH ${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}

# Toggles between CUDA 8 or 9. Needs to be kept in sync with Dockerfile
ENV NCCL_CUDA_VERSION="9"

# Install NCCL2.
RUN wget "https://s3.amazonaws.com/pytorch/nccl_2.1.15-1%2Bcuda${NCCL_CUDA_VERSION}.0_x86_64.txz"
ENV TMP_NCCL_VERSION "nccl_2.1.15-1+cuda${NCCL_CUDA_VERSION}.0_x86_64"
RUN tar -xvf "${TMP_NCCL_VERSION}.txz"
ENV NCCL_ROOT_DIR "$(pwd)/${TMP_NCCL_VERSION}"
ENV LD_LIBRARY_PATH "${NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH}"
RUN rm "${TMP_NCCL_VERSION}.txz"

# Define default command.
CMD ["bash"]
