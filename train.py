import hydra
from pytorch_lightning import Trainer
from omegaconf import DictConfig

from src.model.modules.diffusion import DiffusionModule
from src.data.modules.image import ImageDataModule


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    datamodule = ImageDataModule(cfg.data_dir, cfg.data_loaders)
    model = DiffusionModule()
    trainer = Trainer()

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
