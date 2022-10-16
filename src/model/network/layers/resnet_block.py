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

    def __init__(
        self,
        inplanes: int,
        planes: int,
        t_emb_dim: Optional[int] = None,
        activation: nn.Module = nn.SiLU,
        downsample: Optional[bool] = False,
        upsample: Optional[bool] = False,
    ) -> None:
        super().__init__()

        self.act = activation(inplace=True)
        conv1_stride = 2 if downsample else 1

        self.t_emb_proj = (
            nn.Sequential(self.act, nn.Linear(t_emb_dim, planes))
            if t_emb_dim is not None
            else None
        )
        self.conv1 = conv3x3(inplanes, planes, conv1_stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if downsample:
            self.ds = nn.Sequential(
                conv1x1(inplanes, planes, conv1_stride), nn.BatchNorm2d(planes)
            )
        if upsample:
            self.us_inp = nn.Sequential(
                nn.Upsample(scale_factor=2), conv1x1(inplanes, planes)
            )
            self.us_hidden = nn.Upsample(scale_factor=2)
        self.downsample = downsample
        self.upsample = upsample

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        hidden = x

        # Down/up-sample.
        if self.downsample:
            x = self.ds(x)
        if self.upsample:
            hidden = self.us_hidden(hidden)
            x = self.us_inp(x)

        # First hidden layer.
        hidden = self.conv1(hidden)
        hidden = self.bn1(hidden)
        hidden = self.act(hidden)

        # Inject positional encoding in hidden state.
        if t_emb is not None:
            t_emb = self.t_emb_proj(t_emb)
            hidden = rearrange(t_emb, "b c -> b c 1 1") + hidden

        # Second hidden layer.
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden)

        # Residual connection + activation.
        hidden += x
        hidden = self.act(hidden)

        return hidden
