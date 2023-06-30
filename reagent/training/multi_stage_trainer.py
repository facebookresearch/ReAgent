#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import bisect
import functools
import itertools
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch.nn as nn
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
from reagent.core.utils import lazy_property

from .reagent_lightning_module import ReAgentLightningModule


class MultiStageTrainer(ReAgentLightningModule):
    def __init__(
        self,
        trainers: List[ReAgentLightningModule],
        epochs: List[int],
        assign_reporter_function=None,
        flush_reporter_function=None,
        automatic_optimization: bool = True,
    ) -> None:
        super().__init__(automatic_optimization=automatic_optimization)
        # NB: wrapping in a ModuleList so the state can be saved
        self._trainers = nn.ModuleList(trainers)
        self._assign_reporter_function = assign_reporter_function
        self._flush_reporter_function = (
            functools.partial(flush_reporter_function, self)
            if flush_reporter_function
            else self._flush_reporter
        )
        self._in_testing_loop = False

        # Cumulative sum of number of epochs up to the index (of trainers)
        self._trainer_acc_epochs = [0] + epochs
        for i in range(1, len(epochs) + 1):
            self._trainer_acc_epochs[i] += self._trainer_acc_epochs[i - 1]

        # Num of epochs for each trainer. Used to check if the sum of them
        # equals to num_epochs used in pytorch-lightning trainer
        self.trainer_epoch_mapping = OrderedDict()
        for t, e in zip(trainers, epochs):
            trainer_name = type(t).__name__
            self.trainer_epoch_mapping[trainer_name] = e

    @property
    def multi_stage_total_epochs(self):
        return self._trainer_acc_epochs[-1]

    def set_reporter(self, reporter) -> None:
        super().set_reporter(reporter)
        if self._assign_reporter_function:
            self._assign_reporter_function(self._trainers, reporter)
        else:
            # By default, assume CompoundReporter with the same
            # number of reporters as trainers
            assert len(self._trainers) == len(
                reporter._reporters
            ), f"{len(self._trainers)} != {len(reporter._reporters)}"
            for t, r in zip(self._trainers, reporter._reporters):
                t.set_reporter(r)

    @lazy_property
    def _optimizer_step_to_trainer_idx(self) -> Dict[int, Tuple[int, int]]:
        mapping = {}
        offset = 0

        for i, t in enumerate(self._trainers):
            num_optimizing_steps = t._num_optimizing_steps
            for j in range(num_optimizing_steps):
                mapping[offset + j] = (i, offset)
            offset += num_optimizing_steps

        return mapping

    def _flush_reporter(self, reporter, epoch) -> None:
        """
        By default, assume CompoundReporter with the same
        number of reporters as trainers
        """
        if not self._in_testing_loop:
            epoch_trainer_idx = self._get_trainer_idx_from_epoch()
            reporter._reporters[epoch_trainer_idx].flush(epoch)
        else:
            for r in reporter._reporters:
                r.flush(epoch)

    def on_fit_start(self) -> None:
        self._starting_epoch = self.trainer.current_epoch
        # Connecting pl.Trainer to stage trainers
        for t in self._trainers:
            t.trainer = self.trainer
            t.on_fit_start()

        self.reporter.set_flush_function(self._flush_reporter_function)

    def on_fit_end(self) -> None:
        del self._starting_epoch
        # Disconnecting
        for t in self._trainers:
            t.on_fit_end()
            del t.trainer

        self.reporter.set_flush_function(None)

    def on_test_start(self) -> None:
        self._starting_epoch = self.trainer.current_epoch
        self._in_testing_loop = True

        for t in self._trainers:
            t.on_test_start()

    def on_test_end(self) -> None:
        del self._starting_epoch
        self._in_testing_loop = False
        for t in self._trainers:
            t.on_test_end()

    def _get_trainer_idx_from_epoch(self) -> int:
        # Cycling through the trainers
        epoch = (self.trainer.current_epoch - self._starting_epoch) % (
            self._trainer_acc_epochs[-1]
        )
        trainer_idx = bisect.bisect_right(self._trainer_acc_epochs, epoch) - 1

        return trainer_idx

    def configure_optimizers(self):
        # FIXME: Doesn't support LRScheduler yet
        return list(
            itertools.chain(*[t.configure_optimizers() for t in self._trainers])
        )

    def training_step(self, batch, batch_idx: int, optimizer_idx: int = 0):
        trainer_idx, offset = self._optimizer_step_to_trainer_idx[optimizer_idx]
        epoch_trainer_idx = self._get_trainer_idx_from_epoch()
        assert (
            trainer_idx == epoch_trainer_idx
        ), f"Got {trainer_idx}; expected {epoch_trainer_idx}"
        return self._trainers[trainer_idx].training_step(
            batch, batch_idx, optimizer_idx - offset
        )

    def training_epoch_end(self, outputs) -> None:
        epoch_trainer_idx = self._get_trainer_idx_from_epoch()
        self._trainers[epoch_trainer_idx].training_epoch_end(outputs)

    def validation_step(self, *args, **kwargs):
        epoch_trainer_idx = self._get_trainer_idx_from_epoch()
        return self._trainers[epoch_trainer_idx].validation_step(*args, **kwargs)

    def validation_epoch_end(self, outputs) -> None:
        epoch_trainer_idx = self._get_trainer_idx_from_epoch()
        self._trainers[epoch_trainer_idx].validation_epoch_end(outputs)

    def test_step(self, *args, **kwargs):
        return {
            str(i): trainer.test_step(*args, **kwargs)
            for i, trainer in enumerate(self._trainers)
        }

    def test_epoch_end(self, outputs) -> None:
        for i, trainer in enumerate(self._trainers):
            trainer.test_epoch_end([o[str(i)] for o in outputs])

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: int = False,
        using_native_amp: int = False,
        using_lbfgs: int = False,
    ) -> None:
        assert epoch == self.trainer.current_epoch
        epoch_trainer_idx = self._get_trainer_idx_from_epoch()
        optimizer_trainer_idx, offset = self._optimizer_step_to_trainer_idx[
            optimizer_idx
        ]
        if epoch_trainer_idx == optimizer_trainer_idx:
            # FIXME: epoch argument is not really correct
            # Trainer will see the total epochs, including those epochs they
            # are inactive.
            self._trainers[epoch_trainer_idx].optimizer_step(
                epoch,
                batch_idx,
                optimizer,
                optimizer_idx - offset,
                optimizer_closure,
                on_tpu=on_tpu,
                using_native_amp=using_native_amp,
                using_lbfgs=using_lbfgs,
            )
        # FIXME: this is a hack around https://github.com/PyTorchLightning/pytorch-lightning/pull/9360
        # which assumes that the optimizer closure will be consumed per training step invocation
        # however this is not true in the multi-stage trainer as the training step is called for *all* of the
        # optimizers configured under `trainers` even though only one lightning module is active at a given time
        # A more robust solution would be to use manual optimization, where the lightning trainer does no inspection
        # of the optimization closure for further processing
        elif hasattr(optimizer_closure, "_result"):
            optimizer_closure._result = ClosureResult(closure_loss=None)
