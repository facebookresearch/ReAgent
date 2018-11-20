#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional

import torch
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.training.evaluator import Evaluator


logger = logging.getLogger(__name__)

LOSS_REPORT_INTERVAL = 100


class LossReporter:
    def __init__(self):
        self.td_loss: List[float] = []
        self.incoming_td_loss: List[float] = []

        self.reward_loss: List[float] = []
        self.incoming_reward_loss: List[float] = []

        self.loss_report_interval = LOSS_REPORT_INTERVAL

    @property
    def num_batches(self):
        return len(self.td_loss)

    def report(self, td_loss: float, reward_loss: Optional[float] = None):
        self.incoming_td_loss.append(td_loss)
        if reward_loss is not None:
            self.incoming_reward_loss.append(reward_loss)

        if len(self.incoming_td_loss) >= self.loss_report_interval:
            self.flush()

    def flush(self):
        logger.info("Loss on {} batches".format(len(self.incoming_td_loss)))
        print_details = "Loss:\n"

        td_loss = torch.tensor(self.incoming_td_loss)
        SummaryWriterContext.add_histogram("td_loss", td_loss)
        td_loss_mean = float(td_loss.mean())
        SummaryWriterContext.add_scalar("td_loss/mean", td_loss_mean)
        self.td_loss.append(td_loss_mean)
        print_details = print_details + "TD LOSS: {0:.3f}\n".format(td_loss_mean)

        if len(self.incoming_reward_loss) > 0:
            reward_loss = torch.tensor(self.incoming_reward_loss)
            SummaryWriterContext.add_histogram("reward_loss", reward_loss)
            reward_loss_mean = float(reward_loss.mean())
            SummaryWriterContext.add_scalar("reward_loss/mean", reward_loss_mean)
            self.reward_loss.append(reward_loss_mean)
            print_details = print_details + "REWARD LOSS: {0:.3f}\n".format(
                reward_loss_mean
            )

        for print_detail in print_details.split("\n"):
            logger.info(print_detail)

        self.incoming_td_loss.clear()
        self.incoming_reward_loss.clear()

    def get_last_n_td_loss(self, n):
        return self.td_loss[n:]

    def get_recent_td_loss(self):
        return Evaluator.calculate_recent_window_average(
            self.td_loss, Evaluator.RECENT_WINDOW_SIZE, num_entries=1
        )

    def get_recent_reward_loss(self):
        return Evaluator.calculate_recent_window_average(
            self.reward_loss, Evaluator.RECENT_WINDOW_SIZE, num_entries=1
        )
