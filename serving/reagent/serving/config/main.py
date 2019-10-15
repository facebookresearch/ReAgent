#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os

import click
import reagent.serving.config.applications  # noqa
from reagent.serving.config.builder import DECISION_PLANS


@click.command()
@click.option("--app-id", default=None)
@click.option("--config-dir", default=None)
def export(app_id, config_dir):
    if config_dir is None:
        config_parent_path = os.path.join(
            os.path.expanduser("~"), "configerator/raw_configs"
        )
        if os.path.exists(config_parent_path):
            config_dir = os.path.join(config_parent_path, "reagent/serving")
        else:
            config_dir = os.path.join(os.path.expanduser("~"), "reagent/serving")

    if app_id is None:
        for app_id, configs in DECISION_PLANS.items():
            sub_config_dir = os.path.join(config_dir, app_id)
            if not os.path.exists(sub_config_dir):
                os.makedirs(sub_config_dir)
            for config_name, config in configs.items():
                config_file = os.path.join(sub_config_dir, config_name + '.json')
                print(f"{app_id}:{config_name} exported to {config_file}")
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)
    else:
        if app_id not in DECISION_PLANS:
            raise ValueError(f"App id {app_id} does not exist")
        configs = DECISION_PLANS[app_id]
        sub_config_dir = os.path.join(config_dir, app_id)
        if not os.path.exists(sub_config_dir):
            os.makedirs(sub_config_dir)
        for config_name, config in configs.items():
            config_file = os.path.join(sub_config_dir, config_name, ".json")
            print(f"{app_id}:{config_name} exported to {config_file}")
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)


if __name__ == "__main__":
    export()
