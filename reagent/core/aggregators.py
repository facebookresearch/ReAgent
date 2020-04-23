#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import torch
from reagent.core.tracker import Aggregator
from reagent.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)


class TensorAggregator(Aggregator):
    def __call__(self, key: str, values):
        # Ensure that tensor is on cpu before aggregation.
        values = torch.cat(values, dim=0).cpu()
        return super().__call__(key, values)


def _log_histogram_and_mean(log_key, val):
    try:
        SummaryWriterContext.add_histogram(log_key, val)
        SummaryWriterContext.add_scalar(f"{log_key}/mean", val.mean())
    except ValueError:
        logger.warning(
            f"Cannot create histogram for key: {log_key}; "
            "this is likely because you have NULL value in your input; "
            f"value: {val}"
        )
        raise


class TensorBoardHistogramAndMeanAggregator(TensorAggregator):
    def __init__(self, key: str, log_key: str):
        super().__init__(key)
        self.log_key = log_key

    def aggregate(self, values):
        assert len(values.shape) == 1 or (
            len(values.shape) == 2 and values.shape[1] == 1
        ), f"Unexpected shape for {self.key}: {values.shape}"
        _log_histogram_and_mean(self.log_key, values)


class TensorBoardActionHistogramAndMeanAggregator(TensorAggregator):
    def __init__(
        self,
        key: str,
        category: str,
        title: str,
        actions: List[str],
        log_key_prefix: Optional[str] = None,
    ):
        super().__init__(key)
        self.log_key_prefix = log_key_prefix or f"{category}/{title}"
        self.actions = actions
        SummaryWriterContext.add_custom_scalars_multilinechart(
            [f"{self.log_key_prefix}/{action_name}/mean" for action_name in actions],
            category=category,
            title=title,
        )

    def aggregate(self, values):
        if not (len(values.shape) == 2 and values.shape[1] == len(self.actions)):
            raise ValueError(
                "Unexpected shape for {}: {}; actions: {}".format(
                    self.key, values.shape, self.actions
                )
            )

        for i, action in enumerate(self.actions):
            _log_histogram_and_mean(f"{self.log_key_prefix}/{action}", values[:, i])


class TensorBoardActionCountAggregator(TensorAggregator):
    def __init__(self, key: str, title: str, actions: List[str]):
        super().__init__(key)
        self.log_key = f"actions/{title}"
        self.actions = actions
        SummaryWriterContext.add_custom_scalars_multilinechart(
            [f"{self.log_key}/{action_name}" for action_name in actions],
            category="actions",
            title=title,
        )

    def aggregate(self, values):
        for i, action in enumerate(self.actions):
            SummaryWriterContext.add_scalar(
                f"{self.log_key}/{action}", (values == i).sum().item()
            )


class MeanAggregator(TensorAggregator):
    def __init__(self, key: str):
        super().__init__(key)
        self.values: List[float] = []

    def aggregate(self, values):
        mean = values.mean().item()
        logger.info(f"{self.key}: {mean}")
        self.values.append(mean)


class FunctionsByActionAggregator(TensorAggregator):
    """
    Aggregating the input by action, using the given functions. The input is
    assumed to be an `N x D` tensor, where each column is an action, and
    each row is an example. This takes a dictionary of functions so that the
    values only need to be concatenated once.

    Example:

        agg = FunctionByActionAggregator(
            "model_values", ["A", "B], {"mean": torch.mean, "std": torch.std}
        )

        input = torch.tensor([
            [0.9626, 0.7142],
            [0.7216, 0.5426],
            [0.4225, 0.9485],
        ])
        agg(input)
        input2 = torch.tensor([
            [0.0103, 0.0306],
            [0.9846, 0.8373],
            [0.4614, 0.0174],
        ])
        agg(input2)
        print(agg.values)

        {
            "mean": {
                "A": [0.7022, 0.4854],
                "B": [0.7351, 0.2951],
            },
            "std": {
                "A": [0.2706, 0.4876],
                "B": [0.2038, 0.4696],
            }
        }
    """

    def __init__(self, key: str, actions: List[str], fns: Dict[str, Callable]):
        super().__init__(key)
        self.actions = actions
        self.values: Dict[str, Dict[str, List[float]]] = {
            fn: {action: [] for action in self.actions} for fn in fns
        }
        self.fns = fns

    def aggregate(self, values):
        for name, func in self.fns.items():
            aggregated_values = func(values, dim=0)
            for action, value in zip(self.actions, aggregated_values):
                value = value.item()

                self.values[name][action].append(value)

            latest_values = {
                action: values[-1] for action, values in self.values[name].items()
            }
            logger.info(f"{name} {self.key} {latest_values}")


class ActionCountAggregator(TensorAggregator):
    """
    Counting the frequency of each action. Actions are indexed from `0` to
    `len(actions) - 1`. The input is assumed to contain action index.
    """

    def __init__(self, key: str, actions: List[str]):
        super().__init__(key)
        self.actions = actions
        self.values: Dict[str, List[int]] = {action: [] for action in actions}

    def aggregate(self, values):
        for i, action in enumerate(self.actions):
            self.values[action].append((values == i).sum().item())

        latest_count = {action: counts[-1] for action, counts in self.values.items()}
        logger.info(f"{self.key} {latest_count}")

    def get_distributions(self) -> Dict[str, List[float]]:
        """
        Returns the action disributions in each aggregating step
        """
        totals = np.array([sum(counts) for counts in zip(*self.values.values())])
        return {
            action: (np.array(counts) / totals).tolist()
            for action, counts in self.values.items()
        }

    def get_cumulative_distributions(self) -> Dict[str, float]:
        """
        Returns the cumulative distributions in each aggregating step
        """
        totals = sum(sum(counts) for counts in zip(*self.values.values()))
        return {action: sum(counts) / totals for action, counts in self.values.items()}


_RECENT_DEFAULT_SIZE = int(1e6)


class RecentValuesAggregator(TensorAggregator):
    def __init__(self, key: str, size: int = _RECENT_DEFAULT_SIZE):
        super().__init__(key)
        self.values: Deque[float] = deque(maxlen=size)

    def aggregate(self, values):
        flattened = torch.flatten(values).tolist()
        self.values.extend(flattened)
