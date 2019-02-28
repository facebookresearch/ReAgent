#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds Horizon and runs basic tests.

pip uninstall -y horizon

thrift --gen py --out . ml/rl/thrift/core.thrift

# Installing from current directory, any update will be reflected system-wide
pip install -e .
pytest

# Build the spark package
mvn -f preprocessing/pom.xml clean package
