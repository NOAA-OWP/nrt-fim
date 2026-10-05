from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import torch
import xarray as xr
from geocube.api.core import make_geocube
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig, OmegaConf, open_dict
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import Affine

log = logging.getLogger(__name__)


def _set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for reproducibility

    :param seed: seed value from config file
    :return: None
    """
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read(start_time: datetime, end_time: datetime):
    """Reading the zarr stores for Routing and Soil moisture

    Parameters
    ----------
    start_time : datetime
        start time for slicing the zarr store
    end_time : datetime
        end time for slicing the zarr store

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Routing and Soil moisture datasets
    """
    so = {"anon": True, "default_fill_cache": False, "default_cache_type": "none"}

    base_pattern_routing = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr/"
    base_pattern_sm = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr/"

    ds_r = (
        xr.open_dataset(
            base_pattern_routing,
            backend_kwargs={"storage_options": so},
            engine="zarr",
            chunked_array_type="cubed",
        )
        .sel(time=slice(start_time, end_time))
        .qSfcLatRunoff
    )

    ds_sm = (
        xr.open_dataset(
            base_pattern_sm, backend_kwargs={"storage_options": so}, engine="zarr", chunked_array_type="cubed"
        )
        .sel(time=slice(start_time, end_time))
        .SOIL_M
    )

    return ds_r, ds_sm


def save_prediction_image(pred_tensor, epoch, save_dir, statistics, batch=None):
    """
    Plot denormalized prediction with proper colorbar for flood visualization

    Args:
        prediction: Normalized prediction array/tensor
        mean: Mean value used in normalization
        std: Standard deviation used in normalization
        threshold: Threshold value for considering a pixel as flooded
        figsize: Figure size tuple
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    pred_np = pred_tensor.detach().cpu().squeeze().numpy()

    obs_mean = statistics.loc[2]  # mean is at index 2
    obs_std = statistics.loc[3]  # std is at index 3
    pred_np = ((pred_np - 1e-8) * obs_std) + obs_mean

    vmin = np.min(pred_np)
    vmax = np.max(pred_np)

    # Create a figure with one subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    colors = [(1, 1, 1), (0.8, 0.9, 1), (0.6, 0.8, 1), (0.4, 0.65, 1), (0.2, 0.5, 1), (0, 0.3, 0.8)]
    flood_cmap = LinearSegmentedColormap.from_list("flood_cmap", colors)

    # Plot 1: Denormalized prediction with proper colorbar
    im1 = ax1.imshow(pred_np, cmap=flood_cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"Coffeyville Flood Extent [Epoch: {epoch}]")

    cbar1 = fig.colorbar(im1, ax=ax1)

    # Calculate appropriate tick locations
    # Create 5 evenly spaced ticks from vmin to vmax
    tick_count = 5
    ticks = np.linspace(vmin, vmax, tick_count)
    cbar1.set_ticks(ticks)

    # Format tick labels to have consistent decimal places
    tick_labels = [f"{tick:.2f}" for tick in ticks]
    cbar1.set_ticklabels(tick_labels)

    cbar1.set_label("% of pixels flooded")

    filename = f"pred_epoch_{epoch}" + (f"_batch_{batch}" if batch is not None else "") + ".png"
    plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_array_as_gtiff(
    arr_hw: np.ndarray,  # (H, W)
    out_path: str | Path,  # where to write
    transform: Affine | Sequence[float],  # Affine or GDAL 6-tuple
    crs: str | CRS,  # e.g. "EPSG:4326" or rasterio.crs.CRS
    nodata: float | None = None,
    dtype: np.dtype | None = None,
) -> None:
    """Save a single-band array as a GeoTIFF using the provided transform and CRS."""
    out_path = Path(out_path)
    arr = np.asarray(arr_hw)
    if arr.ndim != 2:
        raise ValueError(f"arr_hw must be 2D (H, W), got shape {arr.shape}")

    # Coerce CRS
    crs_out = crs if isinstance(crs, CRS) else CRS.from_string(str(crs))

    # Coerce transform
    if isinstance(transform, Affine):
        tfm = transform
    elif isinstance(transform, tuple | list) and len(transform) == 6:
        tfm = Affine.from_gdal(*transform)  # (a,b,c,d,e,f)
    else:
        raise ValueError("transform must be an Affine or a 6-tuple/list in GDAL order")

    # Fill NaNs with nodata if requested
    if nodata is not None and np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isnan(arr), nodata, arr)

    # Choose output dtype
    if dtype is None:
        dtype = np.float32 if np.issubdtype(arr.dtype, np.floating) else arr.dtype
    arr_out = arr.astype(dtype, copy=False)

    profile = {
        "driver": "GTiff",
        "height": int(arr_out.shape[0]),
        "width": int(arr_out.shape[1]),
        "count": 1,
        "dtype": str(arr_out.dtype),
        "transform": tfm,
        "crs": crs_out,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2,
        "bigtiff": "IF_SAFER",
        "nodata": nodata,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_out, 1)


def rasterize(
    input_vector: str | Path | gpd.GeoDataFrame,
    output_raster: str | Path,
    resolution: int | float,
    attribute: str,
    crs: pyproj.CRS | str | int,
    dtype: np.dtype | str,
    nodata_value: int | float = None,
) -> None:
    """
    A wrapper around xarray/geocube to convert a vector file or geopandas dataframe into a raster where the value comes from the vector's 'attribute'.

    Args:
    :param input_vector: A vector dataset as a file or geopandas GeoDataFrame.
    :param output_raster: Output path for the raster file.
    :param resolution: Resolution of cells.
    :param attribute: Column name for the value to rasterize.
    :param crs: A spatial coordinate reference system either as a pyproj object, 'EPSG:####' string, or integer (e.g., 4326).
    :param dtype: Numpy data type for the raster.
    :param nodata_value:
    :return: Value to use for no data in the raster. Defaults to None.
    """
    ds = make_geocube(
        input_vector, measurements=[attribute], resolution=(-resolution, resolution), output_crs=crs
    )

    if nodata_value is not None:
        ds[attribute] = ds[attribute].rio.write_nodata(nodata_value)

    options = {"compress": "LZW", "tiled": "TRUE", "dtype": dtype}

    ds[attribute].rio.to_raster(output_raster, **options)


def save_state(
    epoch: int,
    generator: torch.Generator,
    mlp: torch.nn.Module,
    optimizer: torch.nn.Module,
    name: str,
    saved_model_path: Path,
) -> None:
    """Save model state

    Parameters
    ----------
    epoch : int
        The epoch number
    mlp : nn.Module
        The MLP model
    optimizer : nn.Module
        The optimizer
    loss_idx_value : int
        The loss index value
    name: str
        The name of the file we're saving
    """
    mlp_state_dict = {key: value.cpu() for key, value in mlp.state_dict().items()}
    cpu_optimizer_state_dict = {}
    for key, value in optimizer.state_dict().items():
        if key == "state":
            cpu_optimizer_state_dict[key] = {}
            for param_key, param_value in value.items():
                cpu_optimizer_state_dict[key][param_key] = {}
                for sub_key, sub_value in param_value.items():
                    if torch.is_tensor(sub_value):
                        cpu_optimizer_state_dict[key][param_key][sub_key] = sub_value.cpu()
                    else:
                        cpu_optimizer_state_dict[key][param_key][sub_key] = sub_value
        elif key == "param_groups":
            cpu_optimizer_state_dict[key] = []
            for param_group in value:
                cpu_param_group = {}
                for param_key, param_value in param_group.items():
                    cpu_param_group[param_key] = param_value
                cpu_optimizer_state_dict[key].append(cpu_param_group)
        else:
            cpu_optimizer_state_dict[key] = value

    state = {
        "model_state_dict": mlp_state_dict,
        "optimizer_state_dict": cpu_optimizer_state_dict,
        "rng_state": torch.get_rng_state(),
        "data_generator_state": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()

    state["epoch"] = epoch + 1

    torch.save(
        state,
        saved_model_path / f"_{name}_epoch_{state['epoch']}.pt",
    )


def merge_rasters(input_paths: list[str], output_path: str) -> None:
    """
    Merge multiple GeoTIFFs into a single continuous raster, ensuring matching CRS, resolution, compression, and dtype (to minimize output size).

    Args:
        input_paths: List of file paths to GeoTIFFs to merge. They must share the same CRS and pixel resolution.
        output_path: Path for the merged output GeoTIFF.
    """
    import rasterio

    # Open all source datasets
    src_files = [rasterio.open(p) for p in input_paths]
    try:
        # Base metadata from the first file
        base_meta = src_files[0].meta.copy()
        # base_crs = base_meta["crs"]
        # base_res = (base_meta["transform"].a, -base_meta["transform"].e)

        # Inherit compression, predictor, and dtype
        out_compress = base_meta.get("compress")
        out_predictor = base_meta.get("predictor")
        out_dtype = base_meta.get("dtype")
        # If inputs are uncompressed, default to LZW for output to save space
        if not out_compress:
            out_compress = "lzw"
            out_predictor = 2  # good for float data

        # Validate CRS and resolution match across inputs

        mosaic, out_trans = merge(src_files)

        # Update metadata for output
        out_meta = base_meta
        out_meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "dtype": out_dtype,
            }
        )
        if out_compress:
            out_meta["compress"] = out_compress
        if out_predictor:
            out_meta["predictor"] = out_predictor

        # Write merged raster, casting to original dtype
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic.astype(out_dtype))
    finally:
        for src in src_files:
            src.close()


def resolve_cfg_path(cfg: DictConfig, dotted: str) -> str | None:
    """
    Resolve a dotted-string path (e.g., "data_sources.base_pattern_precip") in cfg.Returns None if not found.

    :param cfg: configuration file that includes the data_sources paths
    :param dotted: dotted path
    :return: cured path
    """
    cur = cfg
    for part in dotted.split("."):
        if part not in cur:
            return None
        cur = cur[part]
    return str(cur)


def rewrite_cfg(cfg: DictConfig, key: str, items) -> DictConfig:
    """
    If `items` is a dict: deep-merge into cfg[key], supporting nested + dotted keys.

    Otherwise: replace cfg[key] with `items` (scalar/list/etc.).
    """
    # Scalar / list / None => replace outright
    if not isinstance(items, DictConfig):
        with open_dict(cfg):
            cfg[key] = items
        return cfg

    # Dict => deep-merge (create section if missing or not a mapping)
    with open_dict(cfg):
        if not isinstance(cfg.get(key), (dict | DictConfig)):
            cfg[key] = OmegaConf.create({})

    def _apply(path_prefix: str, obj):
        if isinstance(obj, DictConfig):
            for k, v in obj.items():
                new_path = f"{path_prefix}.{k}" if path_prefix else str(k)
                _apply(new_path, v)
        else:
            # merge=True creates intermediate nodes as needed and respects struct mode
            with open_dict(cfg):
                OmegaConf.update(cfg, f"{key}.{path_prefix}", obj, merge=True)

    _apply("", items)
    return cfg


# if __name__ == "__main__":
#     merge_rasters(
#         input_paths=[
#             r"/Users/farshidrahmani/Dataset/F1_MODIS_USA/DFO_3625_From_20100310_to_20100324/DFO_3625_From_20100310_to_201003240000000000-0000014848.tif",
#             r"/Users/farshidrahmani/Dataset/F1_MODIS_USA/DFO_3625_From_20100310_to_20100324/DFO_3625_From_20100310_to_201003240000000000-0000000000.tif",
#         ],
#         output_path=r"/Users/farshidrahmani/Dataset/F1_MODIS_USA/DFO_3625_From_20100310_to_20100324/DFO_3625_From_20100310_to_20100324_merged.tif",
#     )
