#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import errno
import logging
import os

from ml.rl.polyfill.types_lib.path import Path


logger = logging.getLogger(__name__)


def create_path_if_not_exist(path, on_error=None):
    """
    Attempts to create path if it does not exist. If on_error is
    specified, it is called with an exception if one occurs, otherwise
    exception is rethrown.

    >>> import uuid
    >>> import os
    >>> path = os.path.join("/tmp", str(uuid.uuid4()), str(uuid.uuid4()))
    >>> os.path.exists(path)
    False
    >>> create_path_if_not_exist(path)
    >>> os.path.exists(path)
    True
    """
    logger.info("Creating path {0}".format(path))
    try:
        os.makedirs(path)
        os.chmod(path, 0o777)
    except OSError as ex:
        if ex.errno != errno.EEXIST and not os.path.isdir(path):
            if on_error is not None:
                on_error(path, ex)
            else:
                raise


class PosixPath(Path):
    def do_create_dir(self, path):
        create_path_if_not_exist(path)
