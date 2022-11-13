from typing import Any, Callable, Dict, List, Union

import torch
import torch.nn as nn
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.optim import Adam, Optimizer
from torchmetrics.image.fid import FrechetInceptionDistance

from src.utils.conversion import to_uint8


class DiffusionModule(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss: Callable,
        t_min: int = 1,
        t_max: int = 100,
        beta: float = 0.5,
    ) -> None:
        super().__init__()

        self.network = network
        self.loss = loss
        self.fid = FrechetInceptionDistance()

        self.t_min, self.t_max, self.beta = t_min, t_max, beta

        self.save_hyperparameters(ignore=["network", "loss"])

    def training_step(self, batch: Dict, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        imgs = batch["img"]
        num_imgs = imgs.shape[0]

        ts = torch.randint(self.t_min, self.t_max, (num_imgs,))
        alpha_hat = (1 - self.beta) ** ts
        alpha_hat = alpha_hat[:, None, None, None]
        noise = torch.normal(0, 1, imgs.shape)

        # Apply noise to images (N steps forward diffusion in closed form).
        imgs_d = torch.sqrt(alpha_hat) * imgs + torch.sqrt(1 - alpha_hat) * noise

        noise_pred = self.network(imgs_d, ts)

        loss = self.loss(noise_pred, noise)

        self.log("train/loss", loss)

        return {"loss": loss}

    def validation_step(self, batch: Any, *args: Any, **kwargs: Any) -> None:
        imgs = batch["img"]
        num_imgs = imgs.shape[0]

        x = torch.normal(0, 1, imgs.shape)

        for t_step in reversed(range(self.t_max)):
            ts = torch.ones((num_imgs,), dtype=torch.int64) * t_step
            alpha = torch.ones((num_imgs,)) - self.beta
            alpha = alpha[:, None, None, None]

            if t_step == 0:
                z = torch.zeros(imgs.shape)
            else:
                z = torch.normal(0, 1, imgs.shape)

            noise_pred = self.network(x, ts)

            x_sub = x - (1 - alpha) / torch.sqrt(1 - alpha) * noise_pred
            x = 1 / torch.sqrt(alpha) * x_sub + z * self.beta ** 2

        # Update FID metric.
        self.fid.update(to_uint8(imgs), real=True)
        self.fid.update(to_uint8(x), real=False)

        # Log to WandB.
        wandb.log({"Real images": wandb.Image(imgs)})
        wandb.log({"Generated images": wandb.Image(x)})

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        # Compute and reset FID metric.
        fid_value = self.fid.compute()
        self.fid.reset()

        # Log to WandB.
        wandb.log({"FID": fid_value})

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.network.parameters(), lr=1e-3)
