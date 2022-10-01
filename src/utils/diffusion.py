import numpy as np


def forward_diffusion(img: np.ndarray, betas: np.ndarray):
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
