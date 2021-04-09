#!/usr/bin/env python3
import importlib.util
import os


def is_fb_environment():
    if importlib.util.find_spec("fblearner") is not None:
        if not bool(int(os.environ.get("FORCE_OSS_ENVIRONMENT", False))):
            return True
    return False


IS_FB_ENVIRONMENT = is_fb_environment()
