# pyre-unsafe
import torch
from torch import nn


class ResidualWrapper(nn.Module):
    """
    A wrapper block for residual networks. It is used to wrap a single layer of the network.

    Example:
    layers = []
    for layer in layer_generator:
        layers.append(ResidualWrapper(layer))
    model = torch.nn.Sequential(*layers)
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)
