#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


def requirements():
    with open("docker/requirements.txt") as f:
        return f.read()


setup(
    name="horizon",
    version="0.1",
    author="Facebook",
    description=("Facebook RL"),
    long_description=readme(),
    url="https://github.com/facebookresearch/Horizon",
    license="BSD",
    packages=find_packages(),
    install_requires=[],
    dependency_links=[],
)
