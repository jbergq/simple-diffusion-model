from typing import Any, Dict

from torch.optim import Adam
from pytorch_lightning import LightningModule
from pl_bolts.models.vision.unet import UNet


class DiffusionModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.network = UNet(1, 1)

    def training_step(self, batch, *args, **kwargs) -> Dict[str, Any]:
        imgs, labels = batch
        out = self.network(imgs)

        return {"loss": 0}

    def validation_step(self, *args, **kwargs) -> None:
        pass

    def configure_optimizers(self):
        return Adam(self.network.parameters(), lr=1e-3)
