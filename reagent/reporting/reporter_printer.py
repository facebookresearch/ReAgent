#!/usr/bin/env python3

import logging

from reagent.core.aggregators import MeanAggregator

logger = logging.getLogger(__name__)


class ReporterPrinter:
    def line_plot_mean(self, agg: MeanAggregator):
        raise NotImplementedError()
