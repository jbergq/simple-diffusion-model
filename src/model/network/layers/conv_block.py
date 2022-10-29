import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, act: nn.Module = nn.ReLU) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.act = act(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
