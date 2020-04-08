#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


class BanditsDataset(ABC):
    """
    Base class for logged, aka behavior, dataset
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            length of the dataset
        """
        pass

    @abstractmethod
    def __getitem__(self, idx) -> dataclass:
        """
        Args:
            idx: index of the sample

        Returns:
            tuple of features, action, and reward at idx
        """
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        """
        Returns:
            number of features
        """
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """
        Returns:
            number of total possible actions
        """
        pass

    @property
    @abstractmethod
    def features(self) -> torch.Tensor:
        """
        Returns:
            all features in the dataset as numpy array
        """
        pass

    @property
    @abstractmethod
    def actions(self) -> torch.Tensor:
        """
        Returns:
            all actions in the dataset as numpy array
        """
        pass

    @property
    @abstractmethod
    def rewards(self) -> torch.Tensor:
        """
        Returns:
            all rewards in the dataset as numpy array
        """
        pass
