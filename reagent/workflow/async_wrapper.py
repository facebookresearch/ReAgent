#!/usr/bin/env python3

import functools


def AsyncWrapper(**outer_kwargs):
    def async_wrapper_internal(func):
        @functools.wraps(func)
        def async_wrapper_repeat(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper_repeat

    return async_wrapper_internal
