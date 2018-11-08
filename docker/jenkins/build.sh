#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds Docker image.

set -ex

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

USE_GPU=""
if [[ "$image" == *-cuda* ]]; then
  USE_GPU="1"
fi

# Set Jenkins UID and GID if running Jenkins
if [ -n "${JENKINS:-}" ]; then
  JENKINS_UID=$(id -u jenkins)
  JENKINS_GID=$(id -g jenkins)
fi

pushd ..
docker build \
  --no-cache \
  --build-arg "JENKINS=${JENKINS:-}" \
  --build-arg "JENKINS_UID=${JENKINS_UID:-}" \
  --build-arg "JENKINS_GID=${JENKINS_GID:-}" \
  --build-arg "USE_GPU=${USE_GPU}" \
  -f jenkins.Dockerfile
  "$@" \
  .
popd
