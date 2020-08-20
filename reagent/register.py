#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.fb_checker import IS_FB_ENVIRONMENT


if True:  # To prevent auto sorting of inputs
    # Triggering registration to registries
    import reagent.core.result_types  # noqa
    import reagent.reporting.oss_training_reports  # noqa
    from reagent.model_managers.union import *  # noqa

    if IS_FB_ENVIRONMENT:
        import reagent.core.fb.fb_result_types  # noqa

    # Register all unions
    from reagent.core.union import *  # noqa
    from reagent.model_managers.union import *  # noqa
    from reagent.optimizer.union import *  # noqa
    from reagent.publishers.union import *  # noqa
    from reagent.validators.union import *  # noqa

    if IS_FB_ENVIRONMENT:
        from reagent.model_managers.fb.union import *  # noqa
