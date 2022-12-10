import os
from collections.abc import MutableMapping
from copy import deepcopy

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


def flatten(d, parent_key="", sep="."):
    d = deepcopy(d)

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize modules.
    datamodule: LightningDataModule = instantiate(cfg.datamodule)
    model: LightningModule = instantiate(cfg.model)

    # Initialize logger.
    wandb_logger = WandbLogger(name=cfg.name, project=os.environ["CONDA_DEFAULT_ENV"], config=flatten(cfg))

    # Initialize trainer.
    trainer: Trainer = instantiate(cfg.trainer, logger=wandb_logger)

    # Start the training.
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
