#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


class NoRetriesException(Exception):
    pass


class NonRetryableException(NoRetriesException):
    def __init__(self, child_exception, prefix=""):
        prefixed_message = ""
        if prefix:
            prefixed_message = "{}: ".format(prefix)
        message = "{}{}: {}".format(
            prefixed_message, child_exception.__class__.__name__, str(child_exception)
        )
        super(NonRetryableException, self).__init__(message)
        self.child_exception = child_exception


class NonRetryableTypeError(TypeError, NonRetryableException):
    def __init__(self, type_error):
        super(NonRetryableTypeError, self).__init__(type_error)
