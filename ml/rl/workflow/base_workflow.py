#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import time

from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.workflow.helpers import report_training_status
from ml.rl.workflow.preprocess_handler import PreprocessHandler


logger = logging.getLogger(__name__)


class BaseWorkflow:
    def __init__(
        self, preprocess_handler: PreprocessHandler, trainer, evaluator, minibatch_size
    ):
        self.preprocess_handler = preprocess_handler
        self.trainer = trainer
        self.evaluator = evaluator
        self.minibatch_size = minibatch_size

    def train_network(self, train_dataset, eval_dataset, epochs: int):
        num_batches = int(len(train_dataset) / self.minibatch_size)
        logger.info(
            "Read in batch data set of size {} examples. Data split "
            "into {} batches of size {}.".format(
                len(train_dataset), num_batches, self.minibatch_size
            )
        )

        start_time = time.time()
        for epoch in range(epochs):
            train_dataset.reset_iterator()
            batch_idx = -1
            while True:
                batch_idx += 1
                report_training_status(batch_idx, num_batches, epoch, epochs)
                batch = train_dataset.read_batch()
                if batch is None:
                    break
                tdp = self.preprocess_handler.preprocess(batch)

                tdp.set_type(self.trainer.dtype)
                self.trainer.train(tdp)

            if hasattr(self.trainer, "reward_network"):
                # TODO: Add CPE support to DDPG/SAC
                eval_dataset.reset_iterator()
                accumulated_edp = None
                while True:
                    batch = eval_dataset.read_batch()
                    if batch is None:
                        break
                    tdp = self.preprocess_handler.preprocess(batch)
                    tdp.set_type(self.trainer.dtype)
                    edp = EvaluationDataPage.create_from_tdp(tdp, self.trainer)
                    if accumulated_edp is None:
                        accumulated_edp = edp
                    else:
                        accumulated_edp = accumulated_edp.append(edp)
                assert accumulated_edp is not None, "Eval dataset was empty!"
                accumulated_edp = accumulated_edp.compute_values(self.trainer.gamma)

                cpe_start_time = time.time()
                details = self.evaluator.evaluate_post_training(accumulated_edp)
                details.log()
                details.log_to_tensorboard()
                SummaryWriterContext.increase_global_step()
                logger.info(
                    "CPE evaluation took {} seconds.".format(
                        time.time() - cpe_start_time
                    )
                )

        through_put = (len(train_dataset) * epochs) / (time.time() - start_time)
        logger.info(
            "Training finished. Processed ~{} examples / s.".format(round(through_put))
        )
