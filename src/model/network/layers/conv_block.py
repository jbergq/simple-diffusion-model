from typing import Callable

import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """Simple convolutional block: Conv2D -> BatchNorm -> Activation."""

    def __init__(self, in_size: int, out_size: int, activation: Callable = nn.ReLU) -> None:
        """Constructs the ConvBlock.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.act = activation(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
