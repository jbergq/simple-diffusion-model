import torch
from torchvision.datasets import MNIST

from src.utils.diffusion import forward_diffusion


class MNIST(MNIST):
    def __getitem__(self, index: int):
        imgs, _ = super().__getitem__(index)

        sample = {"img": imgs}

        return sample
