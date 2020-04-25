#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import dataclasses
import importlib
import logging
import os
import sys

import click
from ruamel.yaml import YAML


@click.group()
def reagent():
    from reagent import debug_on_error

    debug_on_error.start()

    os.environ["USE_VANILLA_DATACLASS"] = "0"

    # setup logging in Glog format, approximately...
    FORMAT = (
        "%(levelname).1s%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=FORMAT, datefmt="%m%d %H%M%S"
    )


def _load_func_and_config_class(workflow):
    # Module name should be configurable
    module_name, func_name = workflow.rsplit(".", 1)

    module = importlib.import_module(module_name)
    # Function name should be configurable
    func = getattr(module, func_name)

    # Import in here so that logging and override take place first
    from reagent.core.configuration import make_config_class

    @make_config_class(func)
    class ConfigClass:
        pass

    return func, ConfigClass


def select_relevant_params(config_dict, ConfigClass):
    return {
        field.name: config_dict[field.name]
        for field in dataclasses.fields(ConfigClass)
        if field.name in config_dict
    }


@reagent.command(short_help="Run the workflow with config file")
@click.argument("workflow")
@click.argument("config_file", type=click.File("r"))
def run(workflow, config_file):

    func, ConfigClass = _load_func_and_config_class(workflow)

    # print(ConfigClass.__pydantic_model__.schema_json())
    # At this point, we could load config from a JSON file and create an instance of
    # ConfigClass. Then convert that instance to dict (via .asdict()) and apply to
    # the function

    yaml = YAML(typ="safe")
    config_dict = yaml.load(config_file.read())
    assert config_dict is not None, "failed to read yaml file"
    config_dict = select_relevant_params(config_dict, ConfigClass)
    config = ConfigClass(**config_dict)
    func(**config.asdict())


@reagent.command(short_help="Print JSON-schema of the workflow")
@click.argument("workflow")
def print_schema(workflow):
    func, ConfigClass = _load_func_and_config_class(workflow)

    print(ConfigClass.__pydantic_model__.schema_json())


if __name__ == "__main__":
    reagent()
