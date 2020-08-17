#!/usr/bin/env python3

import functools
import importlib


if importlib.util.find_spec("fblearner") is not None:
    import fblearner.flow.api as flow

    class AsyncWrapper:
        def __init__(self, **kwargs):
            self.async_wrapper = flow.flow_async(**kwargs)
            self.type_wrapper = flow.typed()

        def __call__(self, func):
            return self.async_wrapper(self.type_wrapper(func))


else:

    def AsyncWrapper(**outer_kwargs):
        def async_wrapper_internal(func):
            @functools.wraps(func)
            def async_wrapper_repeat(*args, **kwargs):
                return func(*args, **kwargs)

            return async_wrapper_repeat

        return async_wrapper_internal
