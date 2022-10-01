import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.utils.diffusion import forward_diffusion


class MNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = ToTensor()

    def __getitem__(self, index: int):
        imgs, labels = super().__getitem__(index)
        imgs = np.array(imgs).transpose(1, 2, 0)

        if not self.transform is None:
            imgs = self.transform(imgs)

        imgs_d = forward_diffusion(imgs, betas=np.array(0.1))
        sample = {"img": imgs, "img_d": imgs_d}

        return sample
