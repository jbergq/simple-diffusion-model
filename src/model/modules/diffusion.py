from typing import Any, Dict

import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule
from pl_bolts.models.vision.unet import UNet

from src.model.loss.diffusion import DiffusionLoss


class DiffusionModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.network = UNet(1, 1)
        self.loss = torch.nn.MSELoss()

    def training_step(self, batch, *args, **kwargs) -> Dict[str, Any]:
        imgs, imgs_d = batch["img"], batch["img_d"]
        out = self.network(imgs_d)

        loss = self.loss(out, imgs)

        return {"loss": loss}

    def validation_step(self, *args, **kwargs) -> None:
        pass

    def configure_optimizers(self):
        return Adam(self.network.parameters(), lr=1e-3)
