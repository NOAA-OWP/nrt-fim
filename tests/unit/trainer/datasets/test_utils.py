import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import shapely
import torch
from numpy.testing import assert_array_equal
from omegaconf import DictConfig, OmegaConf

from trainer.utils.utils import rasterize


def test_rasterize() -> None:
    """Builds a simple vector to rasterize. Vector is 4x4 with bottom boxes value=1, top left no data, top right = 2.
    Uses a projected CRS (UTM Zone 31N, at lat-lon 0,0).

    Known warning: RuntimeWarning: invalid value encountered in cast
        data = encode_cf_variable(out_data.variable).values.astype
    Output raster confirmed to have all data and set no data value.
    """
    data_dir = Path(__file__).parents[3] / "data"
    output_raster = data_dir / "output.tif"
    prj = 32631  # UTM 31N - NULL ISLAND

    try:
        # create a vector - a 2x2 box near null island (0, 0)
        box = shapely.geometry.box(166022, 0, 166024, 1)
        box_2 = shapely.geometry.box(166023, 1, 166024, 2)
        gdf = gpd.GeoDataFrame(geometry=[box, box_2], data={"value": [1, 2]}, crs=prj)

        rasterize(
            gdf, output_raster, attribute="value", resolution=1, nodata_value=0, crs=prj, dtype=np.uint8
        )

        with rasterio.open(output_raster) as src:
            prof = src.profile
            transform = src.transform

            # affine projection checks pixel resolution
            assert transform[0] == 1
            assert transform[4] == -1
            assert src.crs == pyproj.crs.CRS(f"EPSG:{prj}").to_wkt()

            assert prof["nodata"] == 0
            assert prof["driver"] == "GTiff"
            assert prof["tiled"] is True
            assert prof["compress"] == "lzw"

            # read raster has right values - 1 nodata as 0
            ras = src.read(1)
            assert_array_equal(ras, np.array([[0, 2], [1, 1]]))

    finally:
        os.remove(output_raster)


def load_trained_config(run_dir: Path) -> DictConfig:
    """
    loads a trained config to evaluate

    :param run_dir: the absolute path to the trained instance run
    :return: the trained config file
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find {cfg_path}")
    return OmegaConf.load(cfg_path)


def find_checkpoint(run_dir: Path, eval_cfg: DictConfig, train_cfg: DictConfig) -> Path:
    """
    finds the checkpoint for an evaluation test

    :param run_dir: the absolute path to the trained instance run
    :param eval_cfg: the evaluation config file
    :param train_cfg: the training config file
    :return: the checkpoint for evaluation
    """

    saved = run_dir / "saved_models"
    if not saved.exists():
        raise FileNotFoundError(f"No saved_models/ in {run_dir}")
    # prefer best_*, else last_*, else any *.pt (most recent by mtime)
    candidates = sorted(saved.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints in {saved}")

    Number = int(eval_cfg.epoch_to_test)
    name = train_cfg.name
    if Number == (-1):
        file_name = candidates[-1]
    elif Number <= int(train_cfg.train.epochs):
        file_name = f"_{name}_epoch_{Number}.pt"
        if not Path(saved / file_name).exists():
            raise FileNotFoundError(f"No checkpoints in {saved}")
    else:
        file_name = candidates[-1]
    checkpoint = Path(saved / file_name)
    return checkpoint


def load_weights(model: torch.nn.Module, ckpt_path: Path, device: str) -> None:
    """
    load the Nn and weights from a checkpoint

    :param model: NN model
    :param ckpt_path: checkpoint path
    :param device: device
    :return: NN model
    """
    blob = torch.load(ckpt_path, map_location=device)
    # handle various checkpoint formats
    if isinstance(blob, dict) and "model_state" in blob:
        state = blob["model_state"]  # BMIBase-style
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]  # common pattern
    elif isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
    else:
        state = blob  # raw state_dict
    return model.load_state_dict(state, strict=False)
