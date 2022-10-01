from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from pytorch_lightning import LightningDataModule

from ..datasets.mnist import MNIST


class ImageDataModule(LightningDataModule):
    def __init__(self, data_dir: str, data_loader_cfg: DictConfig) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.data_loader_cfg = data_loader_cfg

    def setup(self, stage: str):
        mnist_full = MNIST(
            self.data_dir, train=True, transform=ToTensor(), download=True
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, **self.data_loader_cfg.train)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, **self.data_loader_cfg.val)
