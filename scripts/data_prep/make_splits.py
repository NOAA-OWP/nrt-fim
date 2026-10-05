from __future__ import annotations
import hydra
from pathlib import Path
import random
import json
import pandas as pd
from omegaconf import DictConfig
from trainer.utils.utils import _set_seed_everywhere

@hydra.main(
    version_base="1.3",
    config_path="../../config",
    config_name="training_config",
)
def write_splits(cfg: DictConfig):
    """
    Splits dataset into train and test sets

    :param cfg: config file to read fractions and directories
    :return: None. Saves splits files into designated directories
    """
    df = pd.read_csv(cfg.data_sources.index_csv)
    flood_ids = sorted(df["flood_id"].astype(str).unique().tolist())

    _set_seed_everywhere(cfg.seed)
    rnd = random.Random()
    rnd.shuffle(flood_ids)

    train_frac = float(cfg.train.train_frac)

    n = len(flood_ids)
    n_train = int(n * train_frac)
    train_ids = flood_ids[:n_train]
    test_ids  = flood_ids[n_train:]

    splits_dir = Path(cfg.data_sources.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    # write down training IDs
    with open(Path(splits_dir / "train.json"), "w") as json_file:
        json.dump(train_ids, json_file, indent=4)

    #writing down testing IDs
    with open(Path(splits_dir / "eval.json"), "w") as json_file:
        json.dump(test_ids, json_file, indent=4)

    print(f"Wrote splits to {splits_dir}: {len(train_ids)=}, {len(test_ids)=}")

if __name__ == "__main__":
    write_splits()





