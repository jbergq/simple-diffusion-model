from typing import Dict

from torchvision.datasets import MNIST as _MNIST  # type: ignore


class MNIST(_MNIST):
    def __getitem__(self, index: int) -> Dict:
        imgs, _ = super().__getitem__(index)

        sample = {"img": imgs}

        return sample
