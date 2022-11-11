import torch.nn as nn
from torch import Tensor

from src.model.network.layers.conv_block import ConvBlock
from src.model.network.layers.positional_encoding import PositionalEncoding
from src.model.network.layers.resnet_block import ResNetBlockDown, ResNetBlockUp


class UNet(nn.Module):
    """UNet with ResNet blocks and injection of positional encoding."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_layers: int = 5,
        features_start: int = 64,
        t_emb_size: int = 512,
        max_time_steps: int = 1000,
    ) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
            PositionalEncoding(max_time_steps, t_emb_size), nn.Linear(t_emb_size, t_emb_size)
        )

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")
        self.num_layers = num_layers

        self.conv_in = nn.Sequential(ConvBlock(in_size, features_start), ConvBlock(features_start, features_start))

        # Create encoder and decoder stages.
        layers = []
        feats = features_start
        for _ in range(num_layers - 1):  # Encoder
            layers.append(ResNetBlockDown(feats, feats * 2, t_size=t_emb_size))
            feats *= 2
        for _ in range(num_layers - 1):  # Decoder
            layers.append(ResNetBlockUp(feats, feats // 2, skip_size=feats // 2, t_size=t_emb_size))
            feats //= 2
        self.layers = nn.ModuleList(layers)

        self.conv_out = nn.Conv2d(feats, out_size, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        if t is not None:
            # Create time embedding using positional encoding.
            t_emb = self.t_embedding(t)

        x = self.conv_in(x)

        # Store hidden states for U-net skip connections.
        x_i = [x]

        # Encoder stage.
        for layer in self.layers[: self.num_layers - 1]:
            x_i.append(layer(x=x_i[-1], t_emb=t_emb))

        # Decoder stage.
        for i, layer in enumerate(self.layers[self.num_layers - 1 :]):
            x_i[-1] = layer(x=x_i[-1], x_skip=x_i[-2 - i], t_emb=t_emb)

        out = self.conv_out(x_i[-1])

        return out
