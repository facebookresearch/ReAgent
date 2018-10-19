#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

set -e

wget https://archive.apache.org/dist/thrift/0.11.0/thrift-0.11.0.tar.gz -O thrift-0.11.0.tar.gz
tar -xzf thrift-0.11.0.tar.gz
cd thrift-0.11.0
./bootstrap.sh
./configure
make
make install
rm -rf thrift-0.11.0
rm thrift-0.11.0.tar.gz

apt-get install -y openjdk-8-jre maven
