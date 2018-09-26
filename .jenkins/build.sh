#!/bin/bash
# Builds Horizon and runs basic tests.

pip uninstall -y horizon

thrift --gen py --out . ml/rl/thrift/core.thrift

sudo python3 setup.py build develop
python3 setup.py test
