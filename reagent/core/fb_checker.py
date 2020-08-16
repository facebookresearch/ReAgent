#!/usr/bin/env python3
import importlib.util


def is_fb_environment():
    if importlib.util.find_spec("fblearner") is not None:
        return True
    return False


IS_FB_ENVIRONMENT = is_fb_environment()
