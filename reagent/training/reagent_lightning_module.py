#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from reagent.core.utils import lazy_property
from reagent.tensorboardX import SummaryWriterContext


class ReAgentLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._training_step_generator = None
        self._reporter = pl.loggers.base.DummyExperiment()
        self._verified_steps = False
        # For summary_writer property
        self._summary_writer_logger = None
        self._summary_writer = None

    def set_reporter(self, reporter):
        if reporter is None:
            reporter = pl.loggers.base.DummyExperiment()
        self._reporter = reporter
        return self

    @property
    def reporter(self):
        return self._reporter

    def train_step_gen(self, training_batch, batch_idx: int):
        """
        Implement training step as generator here
        """
        raise NotImplementedError

    def soft_update_result(self) -> pl.TrainResult:
        """
        A dummy loss to trigger soft-update
        """
        one = torch.ones(1, requires_grad=True)
        # Create a fake graph to satisfy TrainResult
        # pyre-fixme[16]: Module `pl` has no attribute `TrainResult`.
        return pl.TrainResult(one + one)

    @property
    def summary_writer(self):
        """
        Accessor to TensorBoard's SummaryWriter
        """
        if self._summary_writer_logger is self.logger:
            # If self.logger doesn't change between call, then return cached result
            return self._summary_writer

        # Invalidate
        self._summary_writer = None
        self._summary_writer_logger = self.logger

        if isinstance(self.logger, pl.loggers.base.LoggerCollection):
            for logger in self.logger._logger_iterable:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    self._summary_writer = logger.experiment
                    break
        elif isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
            self._summary_writer = logger.experiment

        return self._summary_writer

    # pyre-fixme[14]: `training_step` overrides method defined in `LightningModule`
    #  inconsistently.
    # pyre-fixme[14]: `training_step` overrides method defined in `LightningModule`
    #  inconsistently.
    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        if self._training_step_generator is None:
            self._training_step_generator = self.train_step_gen(batch, batch_idx)

        ret = next(self._training_step_generator)

        if optimizer_idx == self._num_optimizing_steps - 1:
            if not self._verified_steps:
                try:
                    next(self._training_step_generator)
                except StopIteration:
                    self._verified_steps = True
                if not self._verified_steps:
                    raise RuntimeError("training_step_gen() yields too many times")
            self._training_step_generator = None
            SummaryWriterContext.increase_global_step()

        return ret

    @lazy_property
    def _num_optimizing_steps(self) -> int:
        # pyre-fixme[6]: Expected `Sized` for 1st param but got `Union[None, typing.D...
        return len(self.configure_optimizers())

    def training_epoch_end(self, training_step_outputs):
        # Flush the reporter
        self.reporter.flush(self.current_epoch)
        return pl.TrainResult()
