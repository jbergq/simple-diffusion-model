from typing import Dict

from torchvision.datasets import MNIST as _MNIST

from src.utils.conversion import to_float32


class MNIST(_MNIST):
    def __getitem__(self, index: int) -> Dict:
        imgs, _ = super().__getitem__(index)
        imgs = to_float32(imgs, value_range=(-1.0, 1.0))
        sample = {"img": imgs}

        return sample
