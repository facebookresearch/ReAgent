#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
from datetime import datetime, timedelta

from ml.rl.polyfill.exceptions import NonRetryableTypeError
from ml.rl.polyfill.types_lib.posixpath import PosixPath


def get_root_gluster_path():
    return "/gluster/"


class GlusterPath(PosixPath):
    """
    GlusterPath points to a file on the Gluster file storage system. You can
    expect to read and write to a Gluster file as per the posix api.

    All files created on Gluster through Flow must specify a `retention_period`
    so that we know when to delete this file.
    """

    def __init__(
        self,
        path=None,
        retention_period=None,
        update_retention_period=True,
        needs_backup=True,
        *args,
        **kwargs
    ):
        if not isinstance(retention_period, timedelta):
            raise NonRetryableTypeError("retention_period must be a timedelta")
        super(GlusterPath, self).__init__(path, *args, **kwargs)

    @classmethod
    def _root_dir(cls):
        return os.path.join(
            get_root_gluster_path(), datetime.today().strftime("%Y-%m-%d")
        )
