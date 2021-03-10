#!/usr/bin/env python3

import logging
from typing import Dict

from reagent.core.aggregators import MeanAggregator
from reagent.reporting.reporter_printer import ReporterPrinter
from reagent.tensorboardX import SummaryWriterContext

logger = logging.getLogger(__name__)


class TensorboardReporterPrinter(ReporterPrinter):
    def line_plot_mean(self, agg: MeanAggregator):
        for value in agg.values:
            SummaryWriterContext.add_scalar(f"{agg.key}", value)
