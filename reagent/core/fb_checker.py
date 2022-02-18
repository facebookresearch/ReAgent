#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import importlib.util
import os


def is_fb_environment() -> bool:
    if importlib.util.find_spec("fblearner") is not None:
        if not bool(int(os.environ.get("FORCE_OSS_ENVIRONMENT", False))):
            return True
    return False


IS_FB_ENVIRONMENT: bool = is_fb_environment()
