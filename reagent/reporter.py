#!/usr/bin/env python3


class Reporter:
    """The reporter is responsible for collecting and displaying metrics during training.
    To instrument a trainer, call trainer.set_reporter(...) with an instance of this class.
    Inside a trainer, call reporter.log(metric_a=1, metric_b=2.0, ...).

    In the init function of a reporter, initialize value_list_observers and aggregating_observers and pass them to ReporterBase.
    This defines how metrics are aggregated and plotted.

    Call reporter.log() inside a ReagentLightningModule to log metrics during training.
    """
