#!/usr/bin/env python3

from typing import NamedTuple

from ml.rl.polyfill.types_lib.union import TaggedUnion


class NoPublishingResults(NamedTuple):
    pass


class NoValidationResults(NamedTuple):
    pass


class PublishingResults(TaggedUnion):
    no_publishing_results: NoPublishingResults
    # Add your own validation results type here


class ValidationResults(TaggedUnion):
    no_validation_results: NoValidationResults
    # Add your own validation results type here
