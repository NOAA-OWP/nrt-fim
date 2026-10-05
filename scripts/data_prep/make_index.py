from __future__ import annotations
import hydra
import glob
import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr

from omegaconf import DictConfig
from trainer.data_prep.read_inputs import read_selected_inputs
from trainer.utils.geo_utils import compute_modis_bboxes, filter_modis_paths, extract_floodid


def make_index(cfg: DictConfig) -> pd.DataFrame:
    """
    pre-process target data and build index file for train and test splits

    :param cfg: config file
    :return: None
    """
    inputs_dict = read_selected_inputs(
        cfg=cfg,
        compute_bounds=True
    )
    ref_crs = inputs_dict["input_crs"]
    ref_bounds = inputs_dict["union_bounds"]

    manifest = Path(cfg["data_sources"]["dfo_modis_dir_preprocessed"]) / "manifest.json"
    try:
        # Open the file in read mode ('r')
        with open(manifest, 'r') as json_file:
            # Load the JSON data from the file into a Python dictionary
            pattern = json.load(json_file)

    except FileNotFoundError:
        print(f"Error: The file '{manifest}' was not found.")

    modis_paths = sorted(pattern)
    if not modis_paths:
        raise FileNotFoundError(f"No MODIS TIFF files found in {modis_dir}")

    bboxes = compute_modis_bboxes(modis_paths, input_crs=ref_crs, bounds=ref_bounds)
    kept = filter_modis_paths(modis_paths, bboxes)

    # event_ending_time from filename ..._to_YYYYMMDD_...
    rows = []
    for p in kept:
        name = Path(p).name
        if "_From_" in name:
            start_str = name.split("_From_")[1][:8]
            start_dt = datetime.strptime(start_str, "%Y%m%d")
        else:
            start_dt = None
        if "_to_" in name:
            end_str = name.split("_to_")[1][:8]
            end_dt = datetime.strptime(end_str, "%Y%m%d")
        else:
            end_dt = None
        fid = extract_floodid(p)
        bb = bboxes.get(fid, (None, None, None, None))
        rows.append({
            "flood_id": fid,
            "tif_path": p,
            "start_time": start_dt.isoformat() if start_dt else None,
            "end_time": end_dt.isoformat() if end_dt else None,
            "left": bb[0], "bottom": bb[1], "right": bb[2], "top": bb[3],
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["tif_path"]).reset_index(drop=True)
    return df


@hydra.main(
    version_base="1.3",
    config_path="../../config",
    config_name="training_config",
)
def main(cfg: DictConfig):
    out_csv = Path(cfg.data_sources.index_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = make_index(cfg)
    df.to_csv(out_csv, index=False)
    print(f"Wrote index to {out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    main()
