#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import unittest
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from reagent.reporting import CompoundReporter, ReporterBase
from reagent.training import MultiStageTrainer, ReAgentLightningModule
from torch.utils.data import DataLoader, TensorDataset


class DummyReporter(ReporterBase):
    def __init__(self, name: str, expected_epochs: List[int]):
        super().__init__({}, {})
        self.name = name
        self.expected_epochs = expected_epochs
        self._log_count = 0
        self._flush_count = 0
        self._testing = False

    def log(self, **kwargs) -> None:
        self._log_count += 1

    def flush(self, epoch: int):
        if not self._testing:
            assert epoch in self.expected_epochs, f"{epoch} {self.expected_epochs}"
        self._flush_count += 1


class DummyTrainer(ReAgentLightningModule):
    def __init__(
        self,
        name: str,
        input_dim: int,
        expected_epochs: List[int],
        validation_keys: List[str],
        test_keys: List[str],
    ):
        super().__init__()
        self.name = name
        self.linear1 = nn.Linear(input_dim, 1)
        self.linear2 = nn.Linear(input_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self._call_count = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }
        self.expected_epochs = expected_epochs
        self.validation_keys = validation_keys
        self.test_keys = test_keys

    def configure_optimizers(self):
        return [
            optim.SGD(self.linear1.parameters(), lr=1e2),
            optim.SGD(self.linear2.parameters(), lr=1e2),
        ]

    def on_test_start(self):
        self.reporter._testing = True

    def on_test_end(self):
        self.reporter._testing = False

    def train_step_gen(self, training_batch, batch_idx: int):
        print(f"train_step_gen {self.name}")
        assert (
            self.current_epoch in self.expected_epochs
        ), f"{self.current_epoch} {self.expected_epochs}"
        self._call_count["train"] += 1
        x, label = training_batch

        self.reporter.log()

        y = self.linear1(x)
        yield self.loss_fn(y, label)
        y = self.linear2(x)
        yield self.loss_fn(y, label)

    # pyre-fixme[14]: `validation_step` overrides method defined in
    #  `LightningModule` inconsistently.
    def validation_step(self, batch, batch_idx: int):
        print(f"validation_step {self.name}")
        self._call_count["validation"] += 1
        assert self.current_epoch in self.expected_epochs
        return {k: torch.ones(2, 3) for k in self.validation_keys}

    def validation_epoch_end(self, outputs):
        print(f"validation_step_end {self.name}")
        print(outputs)
        for output in outputs:
            assert set(output.keys()) == set(self.validation_keys)

    # pyre-fixme[14]: `test_step` overrides method defined in `LightningModule`
    #  inconsistently.
    def test_step(self, batch, batch_idx: int):
        print(f"test_step {self.name}")
        self._call_count["test"] += 1
        return {k: torch.ones(2, 3) for k in self.test_keys}

    def test_epoch_end(self, outputs):
        print(f"test_epoch_end {self.name}")
        print(outputs)
        for output in outputs:
            assert set(output.keys()) == set(self.test_keys)


def make_dataset(input_dim, size):
    return TensorDataset(
        torch.randn(size, input_dim),
        torch.randint(0, 2, (size, 1), dtype=torch.float32),
    )


def _merge_report(reporters):
    pass


class TestMultiStageTrainer(unittest.TestCase):
    def test_multi_stage_trainer(self):
        input_dim = 5
        stage1 = DummyTrainer(
            "stage1",
            input_dim,
            expected_epochs=[0, 1, 2],
            validation_keys=["a", "b", "c"],
            test_keys=["d", "e"],
        )
        stage2 = DummyTrainer(
            "stage2",
            input_dim,
            expected_epochs=[3, 4, 5],
            validation_keys=["x", "y", "z"],
            test_keys=["u", "v"],
        )
        multi_stage_trainer = MultiStageTrainer(
            [stage1, stage2],
            epochs=[3, 3],
        )

        reporters = [
            DummyReporter("stage1", expected_epochs=[0, 1, 2]),
            DummyReporter("stage2", expected_epochs=[3, 4, 5]),
        ]
        compound_reporter = CompoundReporter(reporters, _merge_report)
        multi_stage_trainer.set_reporter(compound_reporter)

        training_size = 100
        validation_size = 20
        train_dataloader = DataLoader(
            make_dataset(input_dim, training_size), batch_size=5
        )
        validation_dataloader = DataLoader(
            make_dataset(input_dim, validation_size),
            batch_size=5,
        )

        trainer = pl.Trainer(max_epochs=6, min_epochs=6)
        trainer.fit(multi_stage_trainer, train_dataloader, validation_dataloader)

        test_size = 20
        test_dataloader = DataLoader(
            make_dataset(input_dim, test_size),
            batch_size=5,
        )
        trainer.test(dataloaders=test_dataloader)
        print(f"stage1 {stage1._call_count}")
        print(f"stage2 {stage2._call_count}")
        self.assertEqual(stage1._call_count["train"], 60)
        # It seems that lightning call validation 2 times at the beginning
        self.assertEqual(stage1._call_count["validation"], 14)
        self.assertEqual(stage1._call_count["test"], 4)
        self.assertEqual(stage2._call_count["train"], 60)
        self.assertEqual(stage2._call_count["validation"], 12)
        self.assertEqual(stage2._call_count["test"], 4)

        for reporter, t in zip(reporters, [stage1, stage2]):
            print(f"{reporter.name} {reporter._log_count} {reporter._flush_count}")
            self.assertEqual(reporter._log_count, t._call_count["train"])
            # flush got called in train & validation 3 times each.
            # In stage1, there is an additional call to validation at the beginning
            self.assertEqual(reporter._flush_count, 8 if t == stage1 else 7)
