#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


class HiveDataSet:
    @property
    def namespace(self):
        raise NotImplementedError

    @property
    def tablename(self):
        raise NotImplementedError
