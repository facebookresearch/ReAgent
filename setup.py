#!/usr/bin/env python3

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read()


setup(
    name='horizon',
    version='0.1',
    author='Facebook',
    description=('Facebook RL'),
    long_description=readme(),
    url='https://github.com/facebookresearch/Horizon',
    license='BSD',
    packages=find_packages(),
    install_requires=[],
    dependency_links=[
    ],
    test_suite='ml.rl.test',
)
