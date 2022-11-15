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
    alpha_hat = alpha_hat[..., None, None, None]

    # Apply forward diffusion process in closed form.
    noise = np.random.normal(0, 1, img.shape)
    imgs_noisy = np.sqrt(alpha_hat) * img + np.sqrt(1 - alpha_hat) * noise

    return imgs_noisy


def forward_diffusion_torch(img: Tensor, betas: Tensor) -> Tensor:
    """Applies forward diffusion to the input.

    Number of iterations is determined by length of `betas`.
    """

    assert not torch.any(betas > 1), "Betas must be < 1."

    alphas = 1 - betas
    alpha_hat = torch.prod(alphas)
    alpha_hat = alpha_hat[..., None, None, None]

    # Apply forward diffusion process in closed form.
    noise = torch.normal(0, 1, img.shape)
    imgs_noisy = torch.sqrt(alpha_hat) * img + torch.sqrt(1 - alpha_hat) * noise

    return imgs_noisy
