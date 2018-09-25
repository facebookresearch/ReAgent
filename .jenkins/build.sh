#!/bin/bash
# Builds Horizon and runs basic tests.

pip uninstall -y horizon

thrift --gen py:json --out . ml/rl/thrift/core.thrift
thrift --gen py:json --out . ml/rl/thrift/eval.thrift

sudo python3 setup.py build develop
python3 setup.py test
