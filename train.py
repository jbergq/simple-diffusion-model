import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Instantiate modules and trainer.
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)

    # Start the training.
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
