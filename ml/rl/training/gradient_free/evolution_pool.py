#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List

import torch
import torch.fb.rendezvous.zeus
import torch.nn
import torch.optim
from ml.rl.thrift.core.ttypes import EvolutionParameters


logger = logging.getLogger(__name__)

MAX_RNG_SEED = 100000


class EvolutionPool:
    """
    Handles spawning new individuals from a parent, computing an estimated gradient,
    and applying that gradient to mutate the parent.
    """

    def __init__(
        self,
        seed: int,
        es_params: EvolutionParameters,
        tensor_sizes: Dict[str, List[int]],
    ) -> None:
        self.es_params = es_params
        self.tensor_sizes = tensor_sizes
        self.seed = seed
        assert self.seed < MAX_RNG_SEED, "The random seed must be less than " + str(
            MAX_RNG_SEED
        )
        logger.info("Starting pool with RNG seed: " + str(self.seed))

        # Fill the population with empty values: will populate later
        self.population_tensors: List[Dict[str, torch.Tensor]] = []
        for _ in range(es_params.population_size):
            individual = {}
            for tensor_name, tensor_size in self.tensor_sizes.items():
                individual[tensor_name] = torch.zeros(tensor_size, dtype=torch.float)
            self.population_tensors.append(individual)

        torch.manual_seed(self.seed)
        self.parent_tensors: Dict[str, torch.Tensor] = {}
        for tensor_name, tensor_size in self.tensor_sizes.items():
            self.parent_tensors[tensor_name] = torch.randn(
                tensor_size, dtype=torch.float
            )
            self.parent_tensors[tensor_name].grad = torch.randn(
                tensor_size, dtype=torch.float
            )

        self.optimizer = torch.optim.Adam(
            self.parent_tensors.values(), lr=self.es_params.learning_rate
        )

        self.populate_children(0)

    def populate_children(self, iteration: int):
        torch.manual_seed(iteration * MAX_RNG_SEED + self.seed)
        for individual in self.population_tensors:
            for tensor_name, parent_tensor in self.parent_tensors.items():
                individual_tensor = individual[tensor_name]

                individual_tensor.normal_(0, self.es_params.mutation_power)
                individual_tensor.add_(parent_tensor)

    def apply_global_reward(self, rewards: torch.Tensor, next_iteration: int):
        std_dev = torch.std(rewards)
        if torch.abs(std_dev) > 1e-6:
            normalized_rewards = (rewards - torch.mean(rewards)) / std_dev
            for parent_tensor in self.parent_tensors.values():
                parent_tensor.grad.zero_()
            for i, individual in enumerate(self.population_tensors):
                for tensor_name, parent_tensor in self.parent_tensors.items():
                    individual_tensor = individual[tensor_name]

                    # Subtract the parent to get the gradient estimate
                    individual_tensor.sub_(parent_tensor)

                    # Amplify the gradient by the reward
                    individual_tensor.mul_(normalized_rewards[i])

                    # Divide by a normalizing constant
                    individual_tensor.div_(
                        self.es_params.population_size
                        * self.es_params.mutation_power
                        * -1
                    )

                    parent_tensor.grad += individual_tensor
            self.optimizer.step()

        self.populate_children(next_iteration)

    def compute_all_local_rewards(self):
        return torch.tensor(
            [
                self.compute_local_reward(individual)
                for individual in self.population_tensors
            ],
            dtype=torch.float,
        )

    def compute_local_reward(self, individual):
        """
        Given an individual as a list of tensors, return the reward
        of this policy
        """
        raise NotImplementedError()


class OneMaxEvolutionPool(EvolutionPool):
    """
    A simple example of an evolution pool.  The agent gets maximum reward
    as the tensor approaches [inf, -inf, inf, -inf, ...]
    """

    def compute_local_reward(self, individual):
        sigmoid_params = torch.nn.Sigmoid()(individual["data"])
        total_reward = torch.sum(sigmoid_params[0::2]) + torch.sum(
            1 - sigmoid_params[1::2]
        )
        return total_reward / sigmoid_params.shape[0]
