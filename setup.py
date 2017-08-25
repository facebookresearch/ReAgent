# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rlmodels',
    version='0.1.0',
    description='Reinforcement Learning Model Examples for caffe2',
    long_description=readme,
    author=['Yuchen He', 'Jason Gauci'],
    author_email=['yuchenhe@fb.com', 'jjg@fb.com'],
    url='https://github.com/caffe2/reinforcement-learning-models',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
