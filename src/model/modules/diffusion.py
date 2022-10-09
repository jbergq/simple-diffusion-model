from typing import Any, Dict

import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule


class DiffusionModule(LightningModule):
    def __init__(self, network, loss, t_min=1, t_max=100, beta=0.5) -> None:
        super().__init__()

        self.network = network
        self.loss = loss

        self.t_min, self.t_max, self.beta = t_min, t_max, beta

    def training_step(self, batch, *args, **kwargs) -> Dict[str, Any]:
        imgs = batch["img"]
        num_imgs = imgs.shape[0]

        ts = torch.randint(self.t_min, self.t_max, (num_imgs,))
        alpha_hat = (1 - self.beta) ** ts
        alpha_hat = alpha_hat[:, None, None, None]
        noise = torch.normal(0, 1, imgs.shape)

        # Apply noise to images (N steps forward diffusion in closed form)
        imgs_d = torch.sqrt(alpha_hat) * imgs + torch.sqrt(1 - alpha_hat) * noise

        # TODO(jonathanb): Positional encoding for ts?

        noise_pred = self.network(imgs_d)

        loss = self.loss(noise_pred, noise)

        return {"loss": loss}

    def validation_step(self, *args, **kwargs) -> None:
        pass

    def configure_optimizers(self):
        return Adam(self.network.parameters(), lr=1e-3)
