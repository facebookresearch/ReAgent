#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


class VisualizationMetrics:
    def __init__(self, log_dir):
        # type: (VisualizationMetrics, str) -> None
        self.log_dir = log_dir

    def __repr__(self):
        # type: (VisualizationMetrics) -> str
        return self.log_dir

    __str__ = __repr__

    def __eq__(self, other):
        return type(self) is type(other) and self.log_dir == other.log_dir
