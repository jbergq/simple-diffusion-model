from pytorch_lightning import LightningModule
from pl_bolts.models.vision.unet import UNet


class DiffusionModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.network = UNet()

