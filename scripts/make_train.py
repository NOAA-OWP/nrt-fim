import logging
import time

# import time
from pathlib import Path

import hydra
import torch

# import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# from torch.nn.functional import mse_loss
# from torch.utils.data import DataLoader
#
from trainer import FModel
from trainer._version import __version__
from trainer.datamodules.flood_datamodule import FloodDataModule
from trainer.datasets.registry import compute_in_channels

# from torch.nn.functional import mse_loss
from trainer.loss_functions.mse_loss_func import MSELoss
from trainer.utils.utils import _set_seed_everywhere, save_state

log = logging.getLogger(__name__)
# Quiet only the credential chatter
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
# also quiet broader AWS libs
for name in ("botocore", "boto3", "s3transfer", "s3fs", "fsspec"):
    logging.getLogger(name).setLevel(logging.WARNING)


def training_loop(cfg, nn):
    """
    Building data Module, optimizer, and data loader. Also fixing torch generator

    :param cfg: configuration file that consists of all info needed to train CNN model
    :param nn: The CNN model that has been built in main function
    :return: None - saves the trained model with user-defined intervals
    """
    g = torch.Generator().manual_seed(cfg.seed)

    dm = FloodDataModule(cfg, generator=g)  # << pass it in
    dm.setup(mode="train")
    train_loader = dm.train_dataloader()

    optimizer = torch.optim.Adam(nn.parameters(), lr=cfg.train.lr)

    loss_function = MSELoss(
        ignore_values=[255.0, 1e20],
        large_threshold=1e2,  # values more than 100 don't make sense
        reduction="mean",
    )  # reduced torch array into a single value by talking 'mean' | 'sum' | 'none'
    nn.train()
    for epoch in range(cfg.train.epochs):
        for i, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to(cfg.device), target.to(cfg.device)
            optimizer.zero_grad()
            pred = nn(inputs)
            # loss = mse_loss(pred, target)
            loss_value = loss_function(pred, target, return_stats=False)
            loss_value.backward()
            optimizer.step()
            log.info(f"Epoch {epoch}, sample {i}, Loss: {loss_value.item():.6f}")

        if epoch % int(cfg.train.save_model_interval) == 0:
            save_state(
                epoch=epoch,
                generator=g,
                mlp=nn,
                optimizer=optimizer,
                name=cfg.name,
                saved_model_path=cfg.params.save_path / "saved_models",
            )


@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="training_config",
)
def main(cfg: DictConfig) -> None:
    """
    Using configuration file, setting up randomseeds, building nn, and looping through it.

    :param cfg: Configuration file
    :return: None
    """
    _set_seed_everywhere(cfg.seed)
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)
    (cfg.params.save_path / "saved_models").mkdir(exist_ok=True)
    try:
        start_time = time.perf_counter()
        in_channels = compute_in_channels(cfg)
        nn = FModel(num_classes=int(cfg.model.num_classes), in_channels=in_channels, device=cfg.device)
        training_loop(cfg=cfg, nn=nn)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")

    finally:
        print("Cleaning up...")

        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    print(f"Training F-Model with version: {__version__}")
    main()
