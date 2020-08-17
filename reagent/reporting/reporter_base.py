#!/usr/bin/env python3

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch
from reagent.core import aggregators as agg
from reagent.core.types import RLTrainingOutput


logger = logging.getLogger(__name__)


class ReporterBase:
    def __init__(self, aggregators: List[Tuple[str, agg.Aggregator]]):
        self.aggregators = OrderedDict(aggregators)

    def report(self, **kwargs: Dict[str, Any]):
        for name, value in kwargs.items():
            for aggregator in self.aggregators.values():
                if aggregator.key == name:
                    aggregator.update(name, value)

    def finish_epoch(self):
        for aggregator in self.aggregators.values():
            aggregator.finish_epoch()

    def publish(self) -> RLTrainingOutput:
        pass

    def get_recent(self, key: str, count: int, average: bool):
        for _, aggregator in self.aggregators.items():
            if aggregator.key == key:
                recent = aggregator.aggregator.get_recent(count)
                if len(recent) == 0:
                    return None
                if average:
                    return float(torch.mean(torch.tensor(recent)))
                return recent
        return None

    def get_all(self, key: str, average: bool):
        for _, aggregator in self.aggregators.items():
            if aggregator.key == key:
                all_data = aggregator.aggregator.get_all()
                if len(all_data) == 0:
                    return None
                if average:
                    return float(torch.mean(torch.tensor(all_data)))
                return all_data
        return None

    def __getattr__(self, key: str):
        return self.aggregators[key]

    def end_epoch(self):
        for aggregator in self.aggregators.values():
            aggregator.end_epoch()
