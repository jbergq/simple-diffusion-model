import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Instantiate modules and trainer.
    datamodule: LightningDataModule = instantiate(cfg.datamodule)
    model: LightningModule = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer)

    # Start the training.
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
