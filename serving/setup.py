#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from setuptools import find_packages, setup


def readme():
    print('ok')
    with open("README.md") as f:
        return f.read()


def requirements():
    with open("requirements.txt") as f:
        return f.read()


setup(
    name="ReAgentServing",
    version="0.1",
    author="Facebook",
    description=("ReAgent Serving Platform"),
    long_description=readme(),
    url="https://github.com/facebookresearch/ReAgent",
    license="BSD",
    packages=find_packages(),
    install_requires=[],
    dependency_links=[],
)
