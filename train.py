import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
