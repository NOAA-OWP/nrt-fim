import logging

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from trainer.bmi import FModelBMI
from trainer.datasets.train_dataset import train_dataset
from trainer.utils.utils import save_prediction_image

log = logging.getLogger(__name__)


def main(config_path: str):
    """Main file to run FModel with BMI"""
    # 1) load config.yaml
    cfg = OmegaConf.load(config_path)

    # 2) set up random seed
    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)

    # 3) build test‐only dataset & loader
    ds = train_dataset(cfg, mode="test")
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=False,
    )

    # 4) initialize BMI model
    nn = FModelBMI(cfg)
    nn.initialize()

    for i, mini_batch in enumerate(loader, start=0):
        inputs, target = mini_batch
        inputs, target = inputs.to(cfg.device), target.to(cfg.device)
        pred = nn.update(inputs)

        save_prediction_image(
            pred,
            epoch=0,
            save_dir=cfg.params.save_path / f"output_mb_{i}",
            statistics=ds.target_stats,
            batch=i,
        )

    nn.finalize()


if __name__ == "__main__":
    cfg_path = "../config/training_config.yaml"
    main(cfg_path)
