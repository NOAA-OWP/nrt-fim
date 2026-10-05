from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rasterio import open as rio_open

from trainer.data_prep.read_inputs import read_selected_inputs
from trainer.utils.utils_stats import InputStatsComputer, OnlineStats


def compute_input_stats(cfg: DictConfig) -> dict:
    """
    Checking out whether stats file is already available, otherwise building it

    :param cfg: configuration file
    :return: Statistics of inputs, dynamic and constants combined
    """
    inputs_dict = read_selected_inputs(cfg=cfg, compute_bounds=False)

    # time window for training
    train_start = datetime.strptime(str(cfg.train.start_time), "%Y%m%d")
    train_end = datetime.strptime(str(cfg.train.end_time), "%Y%m%d")
    time_range = (np.datetime64(train_start), np.datetime64(train_end))

    stats_file_name = f"input_stats_{cfg.train.start_time}_{cfg.train.end_time}.json"
    stats_path = Path(cfg.project_root) / "statistics" / stats_file_name

    comp = InputStatsComputer(
        ignore_values=list(cfg["statistics"]["ignore_values_for_inputs"]),
        also_flag_large_as_nodata=bool(cfg["statistics"]["flag_larger_as_nodata_for_inputs"]),
        large_threshold=float(cfg["statistics"]["large_threshold_for_inputs"]),
    )
    stats = comp.load_or_compute_from_inputs_dict(
        inputs_dict=inputs_dict,
        stats_path=stats_path,
        time_range=time_range,
        chunk_size_time=int(cfg.statistics.chunk_size_time),
        chunk_size_y=int(cfg.statistics.chunk_size_y),
        chunk_size_x=int(cfg.statistics.chunk_size_x),
        n_bins=int(cfg.statistics.n_bins),
        calculate_p10_p90=cfg.statistics.calculate_p10_p90,
    )
    return stats


def compute_target_stats(cfg: DictConfig, split_name: str = "train") -> dict:
    """
    Checking out whether stats file is already available, otherwise building it.

    :param cfg: configuration file
    :param split_name: usually set to "train" in common applications
    :return: statistics of satellite images
    """
    df = pd.read_csv(cfg.data_sources.index_csv)
    flood_img_paths = Path(cfg.data_sources.splits_dir, f"{split_name}.json")
    with open(flood_img_paths) as f:
        flood_ids = json.load(f)
    use_ids = sorted(int(s) for s in flood_ids if s.strip())
    modis_paths = df[df.flood_id.isin(use_ids)]["tif_path"].tolist()

    # pull obs sanitation knobs
    ignore_vals = list(cfg.statistics["ignore_values_for_obs"])
    flag_large = bool(cfg.statistics["flag_larger_as_nodata_for_obs"])
    large_thr = float(cfg.statistics["large_threshold_for_obs"])

    # discover band count
    with rio_open(modis_paths[0]) as src0:
        band_count = src0.count

    per_band = [OnlineStats() for _ in range(band_count)]
    for p in modis_paths:
        with rio_open(p) as src:
            for b in range(1, band_count + 1):
                data = src.read(b)
                data_clean = sanitize_1d(
                    data,
                    ignore_values=ignore_vals,
                    flag_larger_as_nodata=flag_large,
                    large_threshold=large_thr,
                )
                per_band[b - 1].update(data_clean.astype("float32"))

    out = {}
    for i, acc in enumerate(per_band, start=1):
        base = acc.finalize()
        # no histogram here; add if you want p10/p90 for targets too
        out[f"band_{i}"] = base

    stats_file_name = f"target_stats_{cfg.train.start_time}_{cfg.train.end_time}.json"
    stats_path = Path(cfg.project_root) / "statistics" / stats_file_name
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"→ Saved target stats to {stats_path}")
    return out


def sanitize_1d(
    arr1d: np.ndarray,
    ignore_values: Iterable[float] | None = (255.0, 1e20),
    flag_larger_as_nodata: bool = True,
    large_threshold: float = 1e19,
) -> np.ndarray:
    """
    Returns a 1D array with nodata sentinels → NaN and drops non-finite.

    Use this for OBS arrays (per-band vectors) before stats.

    """
    a = arr1d.astype("float64", copy=False)

    # mask exact sentinels
    if ignore_values:
        for v in ignore_values:
            a[a == v] = np.nan

    # optionally catch other huge sentinels
    if flag_larger_as_nodata:
        a[np.abs(a) > float(large_threshold)] = np.nan

    # drop non-finite
    a = a[np.isfinite(a)]
    return a


@hydra.main(version_base="1.3", config_path="../../config", config_name="training_config")
def main(cfg: DictConfig):
    """Main function to calculate statistics for inputs and targets"""
    compute_input_stats(cfg)
    compute_target_stats(cfg, "train")
    print("Stats are computed and saved.")


if __name__ == "__main__":
    main()
