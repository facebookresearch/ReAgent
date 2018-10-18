#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
import torch.distributed as distributed
import torch.nn
import torch.optim
from ml.rl.thrift.core.ttypes import EvolutionParameters
from ml.rl.training.gradient_free.evolution_pool import EvolutionPool
from torch.distributed import ProcessGroup


logger = logging.getLogger(__name__)


class EsWorker:
    def __init__(
        self,
        individual_pool: EvolutionPool,
        es_params: EvolutionParameters,
        process_group: ProcessGroup,
        num_nodes: int,
    ) -> None:
        logger.info("WORKER STARTED")

        self.iteration = 0
        self.most_recent_avg_rewards = 0.0
        self.individual_pool = individual_pool
        self.es_params = es_params
        self.process_group = process_group
        self.num_nodes = num_nodes

    def run_epoch(self) -> float:
        logger.info("Computing reward")
        rewards = self.individual_pool.compute_all_local_rewards()
        logger.info("Pushing reward")

        # Sum the rewards across all machines
        distributed.all_reduce(rewards, self.process_group)

        # Divide the rewards by the number of machines.  We do this because
        # there is no "average" all_reduce operator.
        rewards /= self.num_nodes

        self.iteration += 1
        self.individual_pool.apply_global_reward(rewards, self.iteration)
        most_recent_avg_rewards = float(torch.mean(rewards))
        new_parent_reward = self.individual_pool.compute_local_reward(
            self.individual_pool.parent_tensors
        )
        logger.info(
            "ITERATION: {0} MEAN REWARD: {1}, NEW PARENT REWARD: {2}".format(
                self.iteration, most_recent_avg_rewards, new_parent_reward
            )
        )

        return new_parent_reward
