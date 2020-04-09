#!/usr/bin/env python3

from typing import NamedTuple, Optional

from ml.rl.core.dataclasses import dataclass
from ml.rl.core.tagged_union import TaggedUnion


class NoPublishingResults(NamedTuple):
    pass


class NoValidationResults(NamedTuple):
    pass


@dataclass
class PublishingResults(TaggedUnion):
    no_publishing_results: Optional[NoPublishingResults] = None
    # Add your own validation results type here


@dataclass
class ValidationResults(TaggedUnion):
    no_validation_results: Optional[NoValidationResults] = None
    # Add your own validation results type here
