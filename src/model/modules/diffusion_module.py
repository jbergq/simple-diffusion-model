from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import Tensor
from torch.optim import Adam, Optimizer
from torchmetrics.image.fid import FrechetInceptionDistance

from src.utils.conversion import to_image


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

    def _is_last_batch(self, batch_idx: int):
        return batch_idx == self.trainer.num_training_batches - 1

    def training_step(self, batch: Dict, batch_idx: int, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        imgs_real = batch["img"]
        num_imgs = imgs_real.shape[0]

        # Sample random number of iterations to apply noise for in forward diffusion.
        ts = torch.randint(self.t_min, self.t_max, (num_imgs,), device=self.device)
        alpha_hat = (1 - self.beta) ** ts
        alpha_hat = alpha_hat[:, None, None, None]

        # Apply noise to images (N steps forward diffusion in closed form).
        noise = torch.normal(0, 1, imgs_real.shape, device=self.device)
        imgs_noisy = torch.sqrt(alpha_hat) * imgs_real + torch.sqrt(1 - alpha_hat) * noise
        noise_pred = self.network(imgs_noisy, ts)

        loss = self.loss(noise_pred, noise)
        self.log("train/loss", loss)

        if self._is_last_batch(batch_idx):
            # Generate fake images.
            noise = torch.normal(0, 1, imgs_real.shape, device=self.device)
            imgs_fake = self._reverse_diffusion(noise, imgs_real.shape)

            # Rescale images to [0, 255] uint8.
            imgs_real = to_image(imgs_real, value_range=(-1.0, 1.0))
            imgs_fake = to_image(imgs_fake, value_range=(-1.0, 1.0))

            # Log to WandB.
            wandb.log({"train/real images": wandb.Image(imgs_real.float())})
            wandb.log({"train/fake images": wandb.Image(imgs_fake.float())})
            wandb.log({"train/noise images": wandb.Image(imgs_noisy.float())})

        return {"loss": loss}

    def validation_step(self, batch: Any, *args: Any, **kwargs: Any) -> None:
        imgs_real = batch["img"]

        # Generate fake images.
        noise = torch.normal(0, 1, imgs_real.shape, device=self.device)
        imgs_fake = self._reverse_diffusion(noise, imgs_real.shape)

        # Rescale images to [0, 255] uint8.
        imgs_real = to_image(imgs_real, value_range=(-1.0, 1.0))
        imgs_fake = to_image(imgs_fake, value_range=(-1.0, 1.0))

        # Update FID metric.
        self.fid.update(imgs_real, real=True)
        self.fid.update(imgs_fake, real=False)

        # Log to WandB.
        wandb.log({"val/real images": wandb.Image(imgs_real.float())})
        wandb.log({"val/fake images": wandb.Image(imgs_fake.float())})

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        # Compute and reset FID metric.
        fid_value = self.fid.compute()
        self.fid.reset()

        # Log to WandB.
        wandb.log({"val/fid": fid_value})

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.network.parameters(), lr=1e-3)

    def _reverse_diffusion(self, noise: Tensor, shape: Tuple):
        num_imgs = shape[0]

        # Begin reverse diffusion process.
        x = noise

        alpha = torch.ones((num_imgs,), device=self.device) - self.beta
        alpha = alpha[:, None, None, None]
        alpha_hat = alpha.clone()

        with torch.no_grad():
            for t_step in reversed(range(self.t_max)):
                # Create time step tensor injected using model's positional encoding.
                ts = torch.ones((num_imgs,), dtype=torch.int64, device=self.device) * t_step

                # Predict noise for this time step.
                noise_pred = self.network(x, ts)

                # Sample noise to add back for stability.
                if t_step == 0:
                    z = torch.zeros(shape, device=self.device)
                else:
                    z = torch.normal(0, 1, shape, device=self.device)

                # Compute denoised image for this time step.
                x_sub = x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * noise_pred
                x = 1 / torch.sqrt(alpha) * x_sub + self.beta ** 2 * z

                alpha_hat *= alpha

        return x
