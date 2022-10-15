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
