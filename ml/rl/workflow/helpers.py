#!/usr/bin/env python3

import argparse
import json
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def report_training_status(batch_num, num_batches, epoch_num, num_epochs):
    percent_complete = batch_num / (num_batches * num_epochs) * 100
    logger.info(
        "On batch {} of {} and epoch {} of {} ({}% complete)".format(
            batch_num + 1,
            num_batches,
            epoch_num + 1,
            num_epochs,
            round(percent_complete),
        )
    )


def parse_args(args):

    if len(args) != 3:
        raise Exception("Usage: python <file.py> -p <parameters_file>")

    parser = argparse.ArgumentParser(description="Read command line parameters.")
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
    args = parser.parse_args(args[1:])

    with open(args.parameters, "r") as f:
        params = json.load(f)

    return params
