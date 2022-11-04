from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Pad, ToTensor

from pytorch_lightning import LightningDataModule
from src.data.datasets.mnist import MNIST


class ImageDataModule(LightningDataModule):
    def __init__(self, data_dir: str, data_loaders: DictConfig) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.data_loader_cfg = data_loaders

    def setup(self, stage: str) -> None:
        mnist_full = MNIST(
            self.data_dir,
            train=True,
            transform=Compose([Pad(2), ToTensor()]),
            download=True,
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, **self.data_loader_cfg.train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, **self.data_loader_cfg.val)
