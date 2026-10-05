from __future__ import annotations

import logging
from typing import Any

import rioxarray  # noqa: F401
import xarray as xr
from omegaconf import DictConfig
from rasterio.crs import CRS

from trainer.utils.geo_utils import infer_crs

log = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Shared mapping (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
# Feature map: ONLY features listed here can be used in cfg.features.*
#   store : dotted cfg path to the Zarr/GeoTIFF
#   var   : xarray variable name (dynamic only)
#   sel   : optional .sel(...) dict (e.g. layered variables)
#   group : "hourly" | "three_hourly" | "static"
#   stats_key : key in input_stats.json (defaults to var/name)
# ──────────────────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
VAR_NAME_MAP: dict[str, dict[str, Any]] = {
    # hourly dynamics
    "precip": {
        "store": "data_sources.base_pattern_precip",
        "var": "RAINRATE",
        "group": "hourly",
        "stats_key": "precip",
    },
    "air_temp": {
        "store": "data_sources.base_pattern_air_temp",
        "var": "T2D",
        "group": "hourly",
        "stats_key": "air_temp",
    },
    "solar_shortwave": {
        "store": "data_sources.base_pattern_solar_shortwave",
        "var": "SWDOWN",
        "group": "hourly",
        "stats_key": "solar_shortwave",
    },
    # three-hourly dynamics (chrtout, LDAS)
    "terrain_router": {
        "store": "data_sources.base_pattern_routing",
        "var": "sfcheadsubrt",
        "group": "three_hourly",
        "stats_key": "terrain_router",
    },
    "soil_sm_layer1": {
        "store": "data_sources.base_pattern_ldas",
        "var": "SOIL_M",
        "sel": {"soil_layers_stag": 0},
        "group": "three_hourly",
        "stats_key": "soil_sm_layer1",
    },
    "soil_sm_layer2": {
        "store": "data_sources.base_pattern_ldas",
        "var": "SOIL_M",
        "sel": {"soil_layers_stag": 1},
        "group": "three_hourly",
        "stats_key": "soil_sm_layer2",
    },
    "soil_sm_layer3": {
        "store": "data_sources.base_pattern_ldas",
        "var": "SOIL_M",
        "sel": {"soil_layers_stag": 2},
        "group": "three_hourly",
        "stats_key": "soil_sm_layer3",
    },
    "soil_sm_layer4": {
        "store": "data_sources.base_pattern_ldas",
        "var": "SOIL_M",
        "sel": {"soil_layers_stag": 3},
        "group": "three_hourly",
        "stats_key": "soil_sm_layer4",
    },
    "SNOWH": {
        "store": "data_sources.base_pattern_ldas",
        "var": "SNOWH",
        "group": "three_hourly",
        "stats_key": "SNOWH",
    },
    "UGDRNOFF": {
        "store": "data_sources.base_pattern_ldas",
        "var": "UGDRNOFF",
        "group": "three_hourly",
        "stats_key": "UGDRNOFF",
    },
    # statics (GeoTIFFs)
    "flow_accumulation": {
        "store": "data_sources.flow_accumulation",
        "group": "static",
        "stats_key": "flow_accumulation",
    },
    "flow_direction": {
        "store": "data_sources.flow_direction",
        "group": "static",
        "stats_key": "flow_direction",
    },
    "irrigation": {
        "store": "data_sources.irrigation",
        "group": "static",
        "stats_key": "irrigation",
    },
    "surface_extent": {
        "store": "data_sources.surface_extent",
        "group": "static",
        "stats_key": "surface_extent",
    },
}

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────


def resolve_cfg_key(cfg: DictConfig, dotted: str) -> Any:
    """Resolve a dotted key like 'data_sources.base_pattern_precip' inside cfg."""
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def open_feature_da(
    cfg: DictConfig,
    feat: str,
    *,
    storage_options: dict | None = None,
    for_reference: bool = False,
    assume_wgs84_for_static: bool = False,
) -> xr.DataArray | None:
    """
    Open a feature declared in VAR_NAME_MAP as a 2-D DataArray with a CRS.

    - hourly / three_hourly: open Zarr, select var (+sel), if for_reference pick time=0 → 2-D, attach CRS via infer_input_crs
    - static: open GeoTIFF, first band → 2-D, require CRS; optionally write EPSG:4326 if missing

    Returns None on failure.
    """
    meta = VAR_NAME_MAP.get(feat)
    if not meta:
        return None

    group = meta["group"]
    store_key = meta["store"]

    if group in ("hourly", "three_hourly"):
        ds_path = resolve_cfg_key(cfg, store_key)
        so = storage_options or {"anon": True, "default_fill_cache": False, "default_cache_type": "none"}
        try:
            ds = xr.open_dataset(
                ds_path, backend_kwargs={"storage_options": so}, engine="zarr", chunked_array_type="cubed"
            )
            da = ds[meta["var"]]
            if "sel" in meta:
                da = da.sel(**meta["sel"])
            da2d = da.isel(time=0) if ("time" in da.dims and for_reference) else da
            crs = infer_crs(ds)
            # crs = str(ref_da.rio.crs)
            da2d = da2d.rio.write_crs(crs, inplace=False)
            # If not for reference, caller can still slice time later as needed.
            # We always return a DataArray with CRS.
            return da2d
        except FileNotFoundError as e:
            log.warning(f"{ds_path} was not found as {e}")
            return None

    if group == "static":
        tif_path = resolve_cfg_key(cfg, store_key)
        try:
            da = xr.open_dataarray(tif_path, engine="rasterio")  # works fine with rioxarray too
            da2d = da.isel(band=0) if "band" in da.dims else da[0]
            if da2d.rio.crs is None:
                if assume_wgs84_for_static:
                    da2d = da2d.rio.write_crs(CRS.from_epsg(4326), inplace=False)
                else:
                    return None
            return da2d
        except FileNotFoundError:
            log.warning(f"{tif_path} was not found")
            return None

    return None


def compute_in_channels(cfg: DictConfig) -> int:
    """
    Compute the model's in_channels given cfg.features and cfg.train.rho.

    Formula:
      in_channels = len(features.hourly)      * rho
                  + len(features.three_hourly) * (rho // 3)
                  + len(features.static)

    Notes
    -----
      - If a feature list is missing or null or [], it contributes 0.
      - If rho % 3 != 0, we floor-divide for 3-hourly and optionally warn.
      - Raises if result is 0 (no inputs selected).
    """
    hourly = cfg.features.get("hourly") or []
    thrly = cfg.features.get("three_hourly") or []
    stat = cfg.features.get("static") or []

    rho = int(cfg.train.rho)
    if rho <= 0:
        raise ValueError(f"cfg.train.rho must be > 0, got {rho}")
    rho3 = max(1, rho // 3)
    if (rho % 3) != 0 and log is not None:
        log.warning(
            f"rho ({rho}) is not divisible by 3; using floor(rho/3)={rho3} for three-hourly features."
        )

    in_channels = len(hourly) * rho + len(thrly) * rho3 + len(stat)
    if in_channels <= 0:
        raise ValueError(
            "Computed in_channels is 0. Did you disable all features? "
            "Set cfg.features.hourly/three_hourly/static to include at least one feature."
        )
    return in_channels
