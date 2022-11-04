from typing import Union

import numpy as np
import torch
from torch import Tensor

U = Union[Tensor, np.ndarray]


def forward_diffusion(img: U, betas: U) -> U:
    if isinstance(img, Tensor) and isinstance(betas, Tensor):
        return forward_diffusion_torch(img, betas)
    elif isinstance(img, np.ndarray) and isinstance(betas, np.ndarray):
        return forward_diffusion_np(img, betas)
    else:
        raise ValueError("`img` and `betas` must be either torch Tensor or numpy ndarray")


def forward_diffusion_np(img: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Applies forward diffusion to the input.

    Number of iterations is determined by length of `betas`.
    """

    assert not np.any(betas > 1), "Betas must be < 1."

    alphas = 1 - betas
    alpha_hat = np.prod(alphas)
    i_mat = np.identity(img.shape[-1])

    # Apply forward diffusion process in closed form.
    img_d = np.random.normal(np.sqrt(alpha_hat) * img, np.sqrt(1 - alpha_hat) * i_mat)

    return img_d


def forward_diffusion_torch(img: Tensor, betas: Tensor) -> Tensor:
    """Applies forward diffusion to the input.

    Number of iterations is determined by length of `betas`.
    """

    assert not torch.any(betas > 1), "Betas must be < 1."

    alphas = 1 - betas
    alpha_hat = torch.prod(alphas)
    i_mat = torch.eye(img.shape[-1])

    # Apply forward diffusion process in closed form.
    img_d = torch.normal(torch.sqrt(alpha_hat) * img, torch.sqrt(1 - alpha_hat) * i_mat)

    return img_d
