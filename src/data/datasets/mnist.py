import torch
from torchvision.datasets import MNIST

from src.utils.diffusion import forward_diffusion


class MNIST(MNIST):
    def __getitem__(self, index: int):
        imgs, _ = super().__getitem__(index)
        imgs_d = forward_diffusion(imgs, betas=torch.Tensor([0.1]))

        sample = {"img": imgs, "img_d": imgs_d}

        return sample
