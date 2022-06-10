#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class LocalCacheLogger:
    @staticmethod
    def store_metrics(
        tb_logger,
        metrics: Dict[
            str, Union[float, torch.Tensor, Dict[str, Union[float, torch.Tensor]]]
        ],
        step: Optional[int] = None,
    ) -> None:
        for plot_name, plot_value_or_dict in metrics.items():
            if isinstance(plot_value_or_dict, dict):
                if plot_name not in tb_logger.line_plot_buffer:
                    tb_logger.line_plot_buffer[plot_name] = {}
                for line_name, plot_value in plot_value_or_dict.items():
                    LocalCacheLogger._add_point(
                        tb_logger, plot_name, line_name, plot_value, step
                    )
            else:
                LocalCacheLogger._add_point(
                    tb_logger, plot_name, "", plot_value_or_dict, step
                )

    @staticmethod
    def _add_point(
        tb_logger,
        plot_name: str,
        line_name: str,
        plot_value: Union[float, torch.Tensor],
        step: Optional[int],
    ) -> None:
        """Adds a point to a multi-line plot given the plot name, the line name, and optionally the step (x coordinate)."""
        if isinstance(plot_value, torch.Tensor):
            plot_value = plot_value.item()

        if step is None:
            if (
                plot_name in tb_logger.line_plot_buffer
                and line_name in tb_logger.line_plot_buffer[plot_name]
            ):
                x = tb_logger.line_plot_buffer[plot_name][line_name][-1][0] + 1.0
            else:
                x = 0.0
        else:
            x = float(step)

        LocalCacheLogger._create_plots_and_append(
            tb_logger.line_plot_buffer, plot_name, line_name, x, plot_value
        )

        if len(tb_logger.line_plot_buffer[plot_name][line_name]) >= 50:
            mean = float(
                torch.mean(
                    torch.FloatTensor(
                        [
                            float(p[1])
                            for p in tb_logger.line_plot_buffer[plot_name][line_name]
                        ]
                    )
                ).item()
            )
            LocalCacheLogger._create_plots_and_append(
                tb_logger.line_plot_aggregated, plot_name, line_name, x, mean
            )
            tb_logger.line_plot_buffer[plot_name][line_name].clear()

    @staticmethod
    def _create_plots_and_append(
        plot_store: Dict[str, Dict[str, List[Tuple[float, float]]]],
        plot_name: str,
        line_name: str,
        x: int,
        y: float,
    ) -> None:
        if plot_name in plot_store and line_name in plot_store[plot_name]:
            plot_store[plot_name][line_name].append((x, y))
        elif plot_name in plot_store:
            plot_store[plot_name][line_name] = [(x, y)]
        else:
            plot_store[plot_name] = {line_name: [(x, y)]}


class OssTensorboardLogger(TensorBoardLogger):
    """Wrapper around ManifoldTensorBoardLogger that collects the plot data in memory and can flush to create fblearner plot objects."""

    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            save_dir,
            name,
            version,
            log_graph,
            default_hp_metric,
            prefix,
            **kwargs,
        )
        self.line_plot_aggregated: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self.line_plot_buffer: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Dict[
            str, Union[float, torch.Tensor, Dict[str, Union[float, torch.Tensor]]]
        ],
        step: Optional[int] = None,
    ) -> None:
        """Log a set of metrics. A metric is either a scalar or a set of scalars that will be plotted together"""
        super().log_metrics(metrics, step)
        LocalCacheLogger.store_metrics(self, metrics, step)

    def clear_local_data(self) -> None:
        # We don't call clear here because it's a lot of data and someone else probably owns it
        self.line_plot_aggregated = {}
        self.line_plot_buffer = {}
