#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


def requirements():
    with open("requirements.txt") as f:
        return f.read().split("\n")


setup(
    name="reagent",
    version="0.1",
    author="Facebook",
    description=("Facebook RL"),
    long_description=readme(),
    url="https://github.com/facebookresearch/ReAgent",
    license="BSD",
    packages=find_packages(),
    install_requires=[
        "click==7.0",
        "gym[classic_control,box2d,atari]",
        "numpy==1.17.2",
        "pandas==0.25.0",
        "pydantic==1.4",
        "torch",
        "pyspark==2.4.5",
        "ruamel.yaml==0.15.99",
        "scipy==1.3.1",
        "tensorboard==1.14",
        "scikit-learn==0.20.0",
        "xgboost==0.90",
    ],
    dependency_links=[],
    python_requires=">=3.7",
)
