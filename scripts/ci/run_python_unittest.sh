#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Builds ReAgent and runs basic tests.

set -ex

# Installing from current directory, any update will be reflected system-wide
pip3 install --upgrade pip
pip3 install -e .
tox
