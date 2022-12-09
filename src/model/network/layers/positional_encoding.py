import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Transformer sinusoidal positional encoding."""

    def __init__(self, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        """Constructs the PositionalEncoding.

        Args:
            max_time_steps (int): Number of timesteps that can be uniquely represented by encoding.
            embedding_size (int): Size of returned time embedding.
            n (int, optional): User-defined scalar. Defaults to 10000.
        """
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        # Pre-compute the embedding vector for each possible time step.
        # Store in 2D tensor indexed by time step `t` along 0th axis, with embedding vectors along 1st axis.
        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False).to(device)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t: Tensor) -> Tensor:
        """Returns embedding encoding time step `t`.

        Args:
            t (Tensor): Time step.

        Returns:
            Tensor: Returned position embedding.
        """
        return self.pos_embeddings[t, :]
