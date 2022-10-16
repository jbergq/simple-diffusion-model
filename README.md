# Simple Diffusion model

Implementation of denoising diffusion model described in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

## Implementation

### Positional encoding

The transformer sinusoidal positional encoding is used to encode the time step of each sample.

```
class PositionalEncoding(nn.Module):
    def __init__(self, max_time_steps, embedding_size, n=10000):
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(
            max_time_steps, embedding_size, requires_grad=False
        )
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t):
        return self.pos_embeddings[t, :]
```

The figure below shows the elements of each positional encoding vector plotted along the x-axis for each time step on the y-axis.
![positional encoding](./imgs/pos_enc.png)

### ResNet block, w/ positional encoding injection

The positional encoding is injected with a modified ResNet block.

```
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
```
