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

FROM ubuntu:18.04

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

# Define default command.
CMD ["bash"]
