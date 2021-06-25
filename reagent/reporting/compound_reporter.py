#!/usr/bin/env python3

from typing import List, Callable

from reagent.core.result_registries import TrainingReport

from .reporter_base import ReporterBase


class CompoundReporter(ReporterBase):
    def __init__(
        self,
        reporters: List[ReporterBase],
        merge_function: Callable[[List[ReporterBase]], TrainingReport],
    ):
        super().__init__({}, {})
        self._reporters = reporters
        self._merge_function = merge_function
        self._flush_function = None

    def set_flush_function(self, flush_function):
        self._flush_function = flush_function

    def log(self, **kwargs) -> None:
        raise RuntimeError("You should call log() on this reporter")

    def flush(self, epoch: int):
        if self._flush_function:
            self._flush_function(self, epoch)
        else:
            for reporter in self._reporters:
                reporter.flush(epoch)

    def generate_training_report(self) -> TrainingReport:
        return self._merge_function(self._reporters)
