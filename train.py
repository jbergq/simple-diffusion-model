import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize modules.
    datamodule: LightningDataModule = instantiate(cfg.datamodule)
    model: LightningModule = instantiate(cfg.model)

    # Initialize logger.
    wandb_logger = WandbLogger(name=cfg.name, project=os.environ["CONDA_DEFAULT_ENV"])

    # Initialize trainer.
    trainer: Trainer = instantiate(cfg.trainer, logger=wandb_logger)

    # Start the training.
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
