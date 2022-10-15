from typing import Optional

import torch.nn as nn
from torch import Tensor
from einops import rearrange


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetBlock(nn.Module):
    """ResNet block with injection of positional encoding."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[bool] = False,
        t_emb_dim: Optional[int] = None,
        activation: nn.Module = nn.SiLU,
    ) -> None:
        super().__init__()

        self.act = activation(inplace=True)

        self.t_emb_proj = (
            nn.Sequential(self.act, nn.Linear(t_emb_dim, planes))
            if t_emb_dim is not None
            else None
        )

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ds = nn.Sequential(
            conv1x1(inplanes, planes, stride), nn.BatchNorm2d(planes)
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        if t_emb is not None:
            t_emb = self.t_emb_proj(t_emb)
            out = rearrange(t_emb, "b c -> b c 1 1") + out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.ds(x)

        out += identity
        out = self.act(out)

        return out
