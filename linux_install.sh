#!/usr/bin/env bash

# Create virtualenv and install requirements
virtualenv -p python3 env
. env/bin/activate
pip install -r requirements.txt

# Install thrift and generate python bindings
sudo apt install thrift-compiler
thrift --gen py:json --out . ml/rl/thrift/core.thrift

# Install maven for timeline operator
sudo apt install maven
