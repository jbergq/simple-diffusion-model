from typing import Callable

import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, act: Callable = nn.ReLU) -> None:
        """Simple convolutional block: Conv2D -> BatchNorm -> Activation.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            act (Callable, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.act = act(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
