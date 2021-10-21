#!/usr/bin/env python3

import inspect
import logging

import pytorch_lightning as pl
import torch
from reagent.core.tensorboardX import SummaryWriterContext
from reagent.core.utils import lazy_property
from typing_extensions import final


logger = logging.getLogger(__name__)


class ReAgentLightningModule(pl.LightningModule):
    def __init__(self, automatic_optimization=True):
        super().__init__()
        self._automatic_optimization = automatic_optimization
        self._training_step_generator = None
        self._reporter = pl.loggers.base.DummyExperiment()
        # For the generator API
        self._verified_steps = False
        # For summary_writer property
        self._summary_writer_logger = None
        self._summary_writer = None
        # To enable incremental training
        self.register_buffer("_next_stopping_epoch", None)
        self.register_buffer("_cleanly_stopped", None)
        self._next_stopping_epoch = torch.tensor([-1]).int()
        self._cleanly_stopped = torch.ones(1)
        self._setup_input_type()
        self.batches_processed_this_epoch = 0
        self.all_batches_processed = 0

    def _setup_input_type(self):
        self._training_batch_type = None
        sig = inspect.signature(self.train_step_gen)
        assert "training_batch" in sig.parameters
        param = sig.parameters["training_batch"]
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            return
        if hasattr(annotation, "from_dict"):
            self._training_batch_type = annotation

    def set_reporter(self, reporter):
        if reporter is None:
            reporter = pl.loggers.base.DummyExperiment()
        self._reporter = reporter
        return self

    @property
    def reporter(self):
        return self._reporter

    def set_clean_stop(self, clean_stop: bool):
        self._cleanly_stopped[0] = int(clean_stop)

    def increase_next_stopping_epochs(self, num_epochs: int):
        self._next_stopping_epoch += num_epochs
        self.set_clean_stop(False)
        return self

    def train_step_gen(self, training_batch, batch_idx: int):
        """
        Implement training step as generator here
        """
        raise NotImplementedError

    def soft_update_result(self) -> torch.Tensor:
        """
        A dummy loss to trigger soft-update
        """
        one = torch.ones(1, requires_grad=True)
        return one + one

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
    def training_step(self, batch, batch_idx: int, optimizer_idx: int = 0):
        assert (optimizer_idx == 0) or (self._num_optimizing_steps > 1)

        if self._training_step_generator is None:
            if self._training_batch_type and isinstance(batch, dict):
                batch = self._training_batch_type.from_dict(batch)
            self._training_step_generator = self.train_step_gen(batch, batch_idx)

        ret = next(self._training_step_generator)

        if optimizer_idx == self._num_optimizing_steps - 1:
            if not self._verified_steps:
                try:
                    next(self._training_step_generator)
                except StopIteration:
                    self._verified_steps = True
                if not self._verified_steps:
                    raise RuntimeError(
                        "training_step_gen() yields too many times."
                        "The number of yields should match the number of optimizers,"
                        f" in this case {self._num_optimizing_steps}"
                    )
            self._training_step_generator = None
            SummaryWriterContext.increase_global_step()

        return ret

    def optimizers(self, use_pl_optimizer: bool = True):
        # pyre-fixme[6]: Expected `typing_extensions.Literal[True]` for 1st param
        #  but got `bool`.
        o = super().optimizers(use_pl_optimizer)
        if isinstance(o, list):
            return o
        return [o]

    @lazy_property
    def _num_optimizing_steps(self) -> int:
        return len(self.configure_optimizers())

    @final
    def on_epoch_end(self):
        logger.info(
            f"Finished epoch with {self.batches_processed_this_epoch} batches processed"
        )
        self.batches_processed_this_epoch = 0
        # Flush the reporter which has accumulated data in
        # training/validation/test
        self.reporter.flush(self.current_epoch)

        # Tell the trainer to stop.
        if self.current_epoch == self._next_stopping_epoch.item():
            self.trainer.should_stop = True

    @final
    def on_train_batch_end(self, *args, **kwargs):
        self.batches_processed_this_epoch += 1
        self.all_batches_processed += 1

    @final
    def on_validation_batch_end(self, *args, **kwargs):
        self.batches_processed_this_epoch += 1

    @final
    def on_test_batch_end(self, *args, **kwargs):
        self.batches_processed_this_epoch += 1

    def train(self, *args):
        # trainer.train(batch) was the old, pre-Lightning ReAgent trainer API.
        # make sure that nobody is trying to call trainer.train() this way.
        # trainer.train() or trainer.train(True/False) is allowed - this puts the network into training/eval mode.
        if (len(args) == 0) or ((len(args) == 1) and (isinstance(args[0], bool))):
            super().train(*args)
        else:
            raise NotImplementedError(
                "Method .train() is not used for ReAgent Lightning trainers. Please use .fit() method of the pl.Trainer instead"
            )


class StoppingEpochCallback(pl.Callback):
    """
    We use this callback to control the number of training epochs in incremental
    training. Epoch & step counts are not reset in the checkpoint. If we were to set
    `max_epochs` on the trainer, we would have to keep track of the previous `max_epochs`
    and add to it manually. This keeps the infomation in one place.

    Note that we need to set `_cleanly_stopped` back to True before saving the checkpoint.
    This is done in `ModelManager.save_trainer()`.
    """

    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs

    def on_pretrain_routine_end(self, trainer, pl_module):
        assert isinstance(pl_module, ReAgentLightningModule)
        cleanly_stopped = pl_module._cleanly_stopped.item()
        logger.info(f"cleanly stopped: {cleanly_stopped}")
        if cleanly_stopped:
            pl_module.increase_next_stopping_epochs(self.num_epochs)


def has_test_step_override(trainer_module: ReAgentLightningModule):
    """Detect if a subclass of LightningModule has test_step overridden"""
    return type(trainer_module).test_step != pl.LightningModule.test_step
