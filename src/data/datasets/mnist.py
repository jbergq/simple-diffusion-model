from torchvision.datasets import MNIST


class MNIST(MNIST):
    def __getitem__(self, index: int):
        imgs, _ = super().__getitem__(index)

        sample = {"img": imgs}

        return sample
