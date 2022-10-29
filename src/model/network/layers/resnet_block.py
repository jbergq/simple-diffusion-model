from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange


def conv3x3(
    in_size: int,
    out_size: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_size,
        out_size,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_size: int, out_size: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=False)


class ResNetBlock(nn.Module):
    """ResNet block with injection of positional encoding."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        t_dim: Optional[int] = None,
        activation: nn.Module = nn.SiLU,
        downsample: Optional[bool] = False,
        upsample: Optional[bool] = False,
        skip_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.act = activation(inplace=False)

        self.t_proj = (
            nn.Sequential(self.act, nn.Linear(t_dim, out_size))
            if t_dim is not None
            else None
        )

        conv1_stride = 2 if downsample else 1
        conv1_in_size = in_size + skip_size if upsample else in_size

        if downsample:
            self.ds_skip = nn.Sequential(
                conv1x1(in_size, out_size, conv1_stride), nn.BatchNorm2d(out_size)
            )
        if upsample:
            self.us_skip = nn.Sequential(
                nn.Upsample(scale_factor=2), conv1x1(in_size, out_size)
            )
            self.us_hidden = nn.Upsample(scale_factor=2)

        self.conv1 = conv3x3(conv1_in_size, out_size, conv1_stride)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = conv3x3(out_size, out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.downsample = downsample
        self.upsample = upsample

    def forward(self, x: Tensor, x_enc: Tensor = None, t_emb: Tensor = None) -> Tensor:
        x_skip = x

        # TODO(jonathanb): Move down/upsample out of block?

        # Down/upsample.
        if self.downsample:
            x_skip = self.ds_skip(x_skip)
        if self.upsample:
            x_skip = self.us_skip(x_skip)
            x = self.us_hidden(x)

        # Concatenate with encoder features.
        if x_enc is not None:
            x = torch.cat([x, x_enc], dim=1)

        # First hidden layer.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # Inject positional encoding in hidden state.
        if t_emb is not None:
            t_emb = self.t_proj(t_emb)
            x = rearrange(t_emb, "b c -> b c 1 1") + x

        # Second hidden layer.
        x = self.conv2(x)
        x = self.bn2(x)

        # Residual connection + activation.
        x += x_skip
        out = self.act(x)

        return out
