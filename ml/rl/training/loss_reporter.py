#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import List, NamedTuple, Optional

import numpy as np
import torch
from ml.rl.tensorboardX import SummaryWriterContext
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

LOSS_REPORT_INTERVAL = 100


class BatchStats(NamedTuple):
    td_loss: Optional[torch.Tensor] = None
    reward_loss: Optional[torch.Tensor] = None
    imitator_loss: Optional[torch.Tensor] = None
    logged_actions: Optional[torch.Tensor] = None
    logged_propensities: Optional[torch.Tensor] = None
    logged_rewards: Optional[torch.Tensor] = None
    logged_values: Optional[torch.Tensor] = None
    model_propensities: Optional[torch.Tensor] = None
    model_rewards: Optional[torch.Tensor] = None
    model_values: Optional[torch.Tensor] = None
    model_values_on_logged_actions: Optional[torch.Tensor] = None
    model_action_idxs: Optional[torch.Tensor] = None

    def write_summary(self, actions: List[str]):
        if actions:
            for field, log_key in [
                ("logged_actions", "actions/logged"),
                ("model_action_idxs", "actions/model"),
            ]:
                val = getattr(self, field)
                if val is None:
                    continue
                for i, action in enumerate(actions):
                    SummaryWriterContext.add_scalar(
                        "{}/{}".format(log_key, action), (val == i).sum().item()
                    )

        for field, log_key in [
            ("td_loss", "td_loss"),
            ("imitator_loss", "imitator_loss"),
            ("reward_loss", "reward_loss"),
            ("logged_propensities", "propensities/logged"),
            ("logged_rewards", "reward/logged"),
            ("logged_values", "value/logged"),
            ("model_values_on_logged_actions", "value/model_logged_action"),
        ]:
            val = getattr(self, field)
            if val is None:
                continue
            assert len(val.shape) == 1 or (
                len(val.shape) == 2 and val.shape[1] == 1
            ), "Unexpected shape for {}: {}".format(field, val.shape)
            SummaryWriterContext.add_histogram(log_key, val)
            SummaryWriterContext.add_scalar("{}/mean".format(log_key), val.mean())

        for field, log_key in [
            ("model_propensities", "propensities/model"),
            ("model_rewards", "reward/model"),
            ("model_values", "value/model"),
        ]:
            val = getattr(self, field)
            if val is None:
                continue
            if (
                len(val.shape) == 1 or (len(val.shape) == 2 and val.shape[1] == 1)
            ) and not actions:
                SummaryWriterContext.add_histogram(log_key, val)
                SummaryWriterContext.add_scalar("{}/mean".format(log_key), val.mean())
            elif len(val.shape) == 2 and val.shape[1] == len(actions):
                for i, action in enumerate(actions):
                    SummaryWriterContext.add_histogram(
                        "{}/{}".format(log_key, action), val[:, i]
                    )
                    SummaryWriterContext.add_scalar(
                        "{}/{}/mean".format(log_key, action), val[:, i].mean()
                    )
            else:
                raise ValueError(
                    "Unexpected shape for {}: {}; actions: {}".format(
                        field, val.shape, actions
                    )
                )

    @staticmethod
    def add_custom_scalars(action_names: Optional[List[str]]):
        if not action_names:
            return

        SummaryWriterContext.add_custom_scalars_multilinechart(
            [
                "propensities/model/{}/mean".format(action_name)
                for action_name in action_names
            ],
            category="propensities",
            title="model",
        )
        SummaryWriterContext.add_custom_scalars_multilinechart(
            [
                "propensities/logged/{}/mean".format(action_name)
                for action_name in action_names
            ],
            category="propensities",
            title="logged",
        )
        SummaryWriterContext.add_custom_scalars_multilinechart(
            ["actions/logged/{}".format(action_name) for action_name in action_names],
            category="actions",
            title="logged",
        )
        SummaryWriterContext.add_custom_scalars_multilinechart(
            ["actions/model/{}".format(action_name) for action_name in action_names],
            category="actions",
            title="model",
        )


def merge_tensor_namedtuple_list(l, cls):
    def merge_tensor(f):
        vals = [getattr(e, f) for e in l]
        not_none_vals = [v for v in vals if v is not None]
        assert len(not_none_vals) == 0 or len(not_none_vals) == len(vals)
        if not not_none_vals:
            return None
        with torch.no_grad():
            return torch.cat(not_none_vals, dim=0)

    return cls(**{f: merge_tensor(f) for f in cls._fields})


class StatsByAction(object):
    def __init__(self, actions):
        self.stats = {action: [] for action in actions}

    def append(self, stats):
        for k in stats:
            assert k in self.stats
        for k in self.stats:
            v = stats.get(k, 0)
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.stats[k].append(v)

    def items(self):
        return self.stats.items()

    def __len__(self):
        return len(self.stats)


class LossReporter(object):
    RECENT_WINDOW_SIZE = 100

    def __init__(self, action_names: Optional[List[str]] = None):
        assert action_names is None or len(action_names) > 0
        self.action_names: List[str] = action_names or []

        self.td_loss: List[float] = []
        self.reward_loss: List[float] = []
        self.imitator_loss: List[float] = []
        self.logged_action_q_value: List[float] = []
        self.logged_action_counts = {action: 0 for action in self.action_names}
        self.model_values = StatsByAction(self.action_names)
        self.model_value_stds = StatsByAction(self.action_names)
        self.model_action_counts = StatsByAction(self.action_names)
        self.model_action_counts_cumulative = {
            action: 0 for action in self.action_names
        }
        self.model_action_distr = StatsByAction(self.action_names)

        self.incoming_stats: List[BatchStats] = []

        self.loss_report_interval = LOSS_REPORT_INTERVAL

        BatchStats.add_custom_scalars(action_names)

    @property
    def num_batches(self):
        return len(self.td_loss)

    def report(self, **kwargs):
        def _to_tensor(v):
            if v is None:
                return None
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if len(v.shape) == 0:
                v = v.reshape(1)
            return v.detach().cpu()

        kwargs = {k: _to_tensor(v) for k, v in kwargs.items()}
        batch_stats = BatchStats(**kwargs)
        self.incoming_stats.append(batch_stats)
        if len(self.incoming_stats) >= self.loss_report_interval:
            self.flush()

    def flush(self):
        if not len(self.incoming_stats):
            logger.info("Nothing to report")
            return

        logger.info("Loss on {} batches".format(len(self.incoming_stats)))

        batch_stats = merge_tensor_namedtuple_list(self.incoming_stats, BatchStats)
        batch_stats.write_summary(self.action_names)

        print_details = "Loss:\n"

        td_loss_mean = float(batch_stats.td_loss.mean())
        self.td_loss.append(td_loss_mean)
        print_details = print_details + "TD LOSS: {0:.3f}\n".format(td_loss_mean)

        if batch_stats.reward_loss is not None:
            reward_loss_mean = float(batch_stats.reward_loss.mean())
            self.reward_loss.append(reward_loss_mean)
            print_details = print_details + "REWARD LOSS: {0:.3f}\n".format(
                reward_loss_mean
            )

        if batch_stats.imitator_loss is not None:
            imitator_loss_mean = float(batch_stats.imitator_loss.mean())
            self.imitator_loss.append(imitator_loss_mean)
            print_details = print_details + "IMITATOR LOSS: {0:.3f}\n".format(
                imitator_loss_mean
            )

        if batch_stats.model_values is not None and self.action_names:
            self.model_values.append(
                dict(zip(self.action_names, batch_stats.model_values.mean(dim=0)))
            )
            self.model_value_stds.append(
                dict(zip(self.action_names, batch_stats.model_values.std(dim=0)))
            )

        if batch_stats.model_values_on_logged_actions is not None:
            self.logged_action_q_value.append(
                batch_stats.model_values_on_logged_actions.mean().item()
            )

        if (
            batch_stats.logged_actions is not None
            and batch_stats.model_action_idxs is not None
        ):
            logged_action_counts = {
                action: (batch_stats.logged_actions == i).sum().item()
                for i, action in enumerate(self.action_names)
            }
            model_action_counts = {
                action: (batch_stats.model_action_idxs == i).sum().item()
                for i, action in enumerate(self.action_names)
            }
            print_details += "The distribution of logged actions : {}\n".format(
                logged_action_counts
            )
            print_details += "The distribution of model actions : {}\n".format(
                model_action_counts
            )
            for action, count in logged_action_counts.items():
                self.logged_action_counts[action] += count

            self.model_action_counts.append(model_action_counts)

            for action, count in model_action_counts.items():
                self.model_action_counts_cumulative[action] += count

            total = float(sum(model_action_counts.values()))
            self.model_action_distr.append(
                {action: count / total for action, count in model_action_counts.items()}
            )

        print_details += "Batch Evaluator Finished"
        for print_detail in print_details.split("\n"):
            logger.info(print_detail)

        self.incoming_stats.clear()

    def get_last_n_td_loss(self, n):
        return self.td_loss[n:]

    def get_recent_td_loss(self):
        return LossReporter.calculate_recent_window_average(
            self.td_loss, LossReporter.RECENT_WINDOW_SIZE, num_entries=1
        )

    def get_recent_reward_loss(self):
        return LossReporter.calculate_recent_window_average(
            self.reward_loss, LossReporter.RECENT_WINDOW_SIZE, num_entries=1
        )

    def get_recent_imitator_loss(self):
        return LossReporter.calculate_recent_window_average(
            self.imitator_loss, LossReporter.RECENT_WINDOW_SIZE, num_entries=1
        )

    def get_logged_action_distribution(self):
        total_actions = 1.0 * sum(self.logged_action_counts.values())
        return {k: (v / total_actions) for k, v in self.logged_action_counts.items()}

    def get_model_action_distribution(self):
        total_actions = 1.0 * sum(self.model_action_counts_cumulative.values())
        return {
            k: (v / total_actions)
            for k, v in self.model_action_counts_cumulative.items()
        }

    def log_to_tensorboard(self, epoch: int) -> None:
        def none_to_zero(x: Optional[float]) -> float:
            if x is None or math.isnan(x):
                return 0.0
            return x

        for name, value in [
            ("Training/td_loss", self.get_recent_td_loss()),
            ("Training/reward_loss", self.get_recent_reward_loss()),
            ("Training/imitator_loss", self.get_recent_imitator_loss()),
        ]:
            SummaryWriterContext.add_scalar(name, none_to_zero(value), epoch)

    @staticmethod
    def calculate_recent_window_average(arr, window_size, num_entries):
        if len(arr) > 0:
            begin = max(0, len(arr) - window_size)
            return np.mean(np.array(arr[begin:]), axis=0)
        else:
            logger.error("Not enough samples for evaluation.")
            if num_entries == 1:
                return float("nan")
            else:
                return [float("nan")] * num_entries
