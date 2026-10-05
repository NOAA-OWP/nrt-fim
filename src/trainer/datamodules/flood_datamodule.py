from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler

from trainer.datasets.flood_dataset import FloodDataset


def _seed_worker(worker_id: int):
    """Make numpy & random deterministic per worker"""
    #
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FloodDataModule:
    """a data loader class for training and testing of flood events"""

    def __init__(self, cfg: DictConfig, generator: torch.Generator | None = None):
        """
        Initialize config file

        :param cfg: configuration file
        """
        self.cfg = cfg
        self.generator = generator or torch.Generator().manual_seed(int(cfg.seed))
        self.train_ds = None
        self.eval_ds = None

    def setup(self, mode: list | str):
        """
        Sets up datasets

        :return: None
        """
        if "train" in mode:
            self.train_ds = FloodDataset(self.cfg, split="train")
        if ("eval" in mode) or ("user_defined" in mode):
            self.eval_ds = FloodDataset(self.cfg, split="eval")

    def train_dataloader(
        self,
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
        steps_per_epoch: int | None = None,
        generator: torch.Generator | None = None,
    ):
        """
        An overridable train dataloader.

        :param batch_size: batch size in each iteration. reads from cfg
        :param num_workers: number of cpus for parallel preparation of the batch. reads from cfg
        :param steps_per_epoch: number of samples per epoch equals to batch_size * steps_per_epoch
        :param generator: torch random seed
        :return: Dataloader
        """
        bs = int(batch_size if batch_size is not None else self.cfg.train.batch_size)
        nw = int(num_workers if num_workers is not None else self.cfg.train.num_workers)
        st = int(steps_per_epoch if steps_per_epoch is not None else self.cfg.train.steps_per_epoch)
        gen = generator or self.generator

        sampler = RandomSampler(self.train_ds, replacement=True, num_samples=st * bs, generator=gen)
        return DataLoader(
            self.train_ds,
            batch_size=bs,
            sampler=sampler,
            drop_last=True,
            num_workers=nw,
            generator=gen,
            worker_init_fn=_seed_worker,
            persistent_workers=bool(nw > 0),
        )

    def eval_dataloader(
        self,
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
        shuffle: bool | None = None,
        steps: int | None = None,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
    ):
        """
        If `steps` is provided, builds a fixed-length eval using a sampler.

        Otherwise, iterates the dataset once (shuffle=False by default).
        """
        bs = int(batch_size if batch_size is not None else self.cfg.eval.batch_size)
        nw = int(num_workers if num_workers is not None else self.cfg.eval.num_workers)
        shf = bool(False if shuffle is None else shuffle)
        gen = generator or self.generator

        if steps is not None:
            sampler = RandomSampler(
                self.eval_ds, replacement=True, num_samples=int(steps) * bs, generator=gen
            )
            return DataLoader(
                self.eval_ds,
                batch_size=bs,
                sampler=sampler,
                drop_last=drop_last,
                num_workers=nw,
                generator=gen,
                worker_init_fn=_seed_worker,
                persistent_workers=bool(nw > 0),
            )
        else:
            return DataLoader(
                self.eval_ds,
                batch_size=bs,
                shuffle=shf,
                drop_last=drop_last,
                num_workers=nw,
                generator=gen,
                worker_init_fn=_seed_worker,
                persistent_workers=bool(nw > 0),
            )
