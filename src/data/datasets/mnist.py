from torchvision.datasets import MNIST


class MNIST(MNIST):
    def __getitem__(self, index: int):
        sample = super().__getitem__(index)

        return sample
