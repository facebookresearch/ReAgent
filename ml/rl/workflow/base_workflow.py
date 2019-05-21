#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import gzip
import json
import logging
import os
import time
from typing import Dict

import torch
from ml.rl.preprocessing import normalization
from ml.rl.readers.data_streamer import DataStreamer
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.thrift.core.ttypes import NormalizationParameters
from ml.rl.workflow.page_handler import (
    EvaluationPageHandler,
    TrainingPageHandler,
    feed_pages,
)
from ml.rl.workflow.preprocess_handler import PreprocessHandler
from torch import distributed


logger = logging.getLogger(__name__)


class BaseWorkflow:
    def __init__(
        self, preprocess_handler: PreprocessHandler, trainer, evaluator, minibatch_size
    ):
        self.preprocess_handler = preprocess_handler
        self.trainer = trainer
        self.evaluator = evaluator
        self.minibatch_size = minibatch_size

    @staticmethod
    def read_norm_file(path) -> Dict[int, NormalizationParameters]:
        path = os.path.expanduser(path)
        if path.split(".")[-1] == "gz":
            with gzip.open(path) as f:
                norm_json = json.load(f)
        else:
            with open(path) as f:
                norm_json = json.load(f)
        return normalization.deserialize(norm_json)

    @staticmethod
    def init_multiprocessing(
        num_processes_per_node, num_nodes, node_index, gpu_index, init_method
    ):
        assert max(1, torch.cuda.device_count()) == int(
            num_processes_per_node
        ), "Not all nodes have the same number of GPUs!"
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(gpu_index)
        world_size = num_nodes * num_processes_per_node
        rank = (node_index * torch.cuda.device_count()) + gpu_index
        distributed.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=rank
        )

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
            data_streamer = DataStreamer(train_dataset, pin_memory=self.trainer.use_gpu)
            preprocess_handler = self.preprocess_handler
            dtype = self.trainer.dtype

            def preprocess(batch):
                tdp = preprocess_handler.preprocess(batch)
                tdp.set_type(dtype)
                return tdp

            feed_pages(
                data_streamer,
                len(train_dataset),
                epoch,
                self.minibatch_size,
                self.trainer.use_gpu,
                TrainingPageHandler(self.trainer),
                batch_preprocessor=preprocess,
            )

            if hasattr(self.trainer, "q_network_cpe"):
                # TODO: Add CPE support to DDPG/SAC, Parametric DQN (once moved to modular)
                eval_dataset.reset_iterator()
                data_streamer = DataStreamer(
                    eval_dataset, pin_memory=self.trainer.use_gpu
                )
                eval_page_handler = EvaluationPageHandler(
                    self.trainer, self.evaluator, self
                )
                feed_pages(
                    data_streamer,
                    len(eval_dataset),
                    epoch,
                    self.minibatch_size,
                    self.trainer.use_gpu,
                    eval_page_handler,
                    batch_preprocessor=preprocess,
                )

                SummaryWriterContext.increase_global_step()

        through_put = (len(train_dataset) * epochs) / (time.time() - start_time)
        logger.info(
            "Training finished. Processed ~{} examples / s.".format(round(through_put))
        )

    def report(self, evaluation_details):
        evaluation_details.log_to_tensorboard()
