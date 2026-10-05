from __future__ import annotations

import logging
from typing import Any

import rasterio
import xarray as xr
from omegaconf import DictConfig
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.warp import transform_bounds
from rioxarray.exceptions import MissingCRS

from trainer.datasets.registry import VAR_NAME_MAP
from trainer.utils.geo_utils import _resolve_dynamic_da, _to_crs_obj, infer_crs
from trainer.utils.utils import resolve_cfg_path

log = logging.getLogger(__name__)


def read_selected_inputs(
    cfg: DictConfig,
    compute_bounds: bool = False,
    var_name_map: dict[str, dict[str, Any]] = VAR_NAME_MAP,
) -> dict[str, Any]:
    """
    Open only the inputs requested in the config and computes the bounds:

      - Dynamic hourly & 3-hourly variables from Zarr
      - Static rasters (paths only; opened lazily unless compute_bounds=True)

    Returns a dictionary with:
      dyn_ds:           {store_key -> xr.Dataset}
      dyn_vars:         {feature_name -> xr.DataArray}   # renamed to stats_key
      input_crs:        rasterio.CRS (or string in case of fallback)
      hourly_time:      np.ndarray | None
      three_hourly_time:np.ndarray | None
      static_paths:     {feature_name -> str}            # absolute/URL paths
      bounds_per_feature (optional): {feature_name -> (minx, miny, maxx, maxy)} (in input_crs)
      union_bounds (optional): (minx, miny, maxx, maxy)  (in input_crs)
    """
    # Use cfg.features.* if explicit lists not provided
    feats_hourly = sorted(cfg.features.get("hourly") or [])
    feats_threeh = sorted(cfg.features.get("three_hourly") or [])
    feats_static = sorted(cfg.features.get("static") or [])

    # Filter out unknown or empty features defensively
    feats_hourly = [f for f in feats_hourly if f in var_name_map]
    feats_threeh = [f for f in feats_threeh if f in var_name_map]
    feats_static = [f for f in feats_static if f in var_name_map]

    # ── Open dynamic datasets per store ────────────────────────────────────────
    dyn_ds: dict[str, xr.Dataset] = {}
    crs_by_store: dict[str, CRS] = {}
    dyn_vars: dict[str, xr.DataArray] = {}
    hourly_time = None
    three_hourly_time = None

    needed_store_keys = {var_name_map[n]["store"] for n in (feats_hourly + feats_threeh)}
    if needed_store_keys:
        so = {"anon": True, "default_fill_cache": False, "default_cache_type": "none"}
        for store_key in needed_store_keys:
            url = resolve_cfg_path(cfg, store_key)
            if url is None:
                raise KeyError(f"Missing path in cfg for '{store_key}' (required by a selected feature).")
            ds = xr.open_dataset(
                url,
                backend_kwargs={"storage_options": so},
                engine="zarr",
                chunked_array_type="cubed",
            )
            dyn_ds[store_key] = ds
            # infer CRS per store
            try:
                crs_by_store[store_key] = _to_crs_obj(infer_crs(ds))
            except (MissingCRS, CRSError):
                crs_by_store[store_key] = CRS.from_epsg(4326)

        # choose a canonical working CRS (first store’s), warn if mixed
        store_crs_list = list(crs_by_store.values())
        input_crs = store_crs_list[0]

        # materialize variables and attach their **store-specific** CRS if missing
        for name in feats_hourly + feats_threeh:
            store_key = var_name_map[name]["store"]
            da = _resolve_dynamic_da(name, dyn_ds, var_name_map=var_name_map)
            if getattr(da.rio, "crs", None) is None:
                da = da.rio.write_crs(crs_by_store[store_key], inplace=False)
            dyn_vars[name] = da

            # capture time axes from the first variable in each group
            if ("time" in da.dims) and (name in feats_hourly) and (hourly_time is None):
                hourly_time = da["time"].values
            if ("time" in da.dims) and (name in feats_threeh) and (three_hourly_time is None):
                three_hourly_time = da["time"].values
    else:
        # no dynamics selected
        input_crs = CRS.from_epsg(4326)

    # ── Static paths (don’t open heavy rasters unless computing bounds) ────────
    static_paths: dict[str, str] = {}
    for name in feats_static:
        p = resolve_cfg_path(cfg, var_name_map[name]["store"])
        if p:
            static_paths[name] = p

    out: dict[str, Any] = {
        "dyn_ds": dyn_ds,
        "dyn_vars": dyn_vars,
        "crs_by_store": crs_by_store,
        "input_crs": input_crs,
        "hourly_time": hourly_time,
        "three_hourly_time": three_hourly_time,
        "static_paths": static_paths,
    }

    # ── Optional bounds per feature (all reported in input_crs) ───────────────
    if compute_bounds:
        bounds_per_feature: dict[str, tuple[float, float, float, float]] = {}
        union: tuple[float, float, float, float] | None = None

        # dynamic features: compute bounds in their store CRS, transform to input_crs if needed
        for name, da in dyn_vars.items():
            try:
                b = da.rio.bounds()
                store_key = var_name_map[name]["store"]
                da_crs = crs_by_store.get(store_key, input_crs)
                if da_crs != input_crs:
                    b = transform_bounds(da_crs, input_crs, *b)
                bounds_per_feature[name] = b
                union = (
                    b
                    if union is None
                    else (max(union[0], b[0]), max(union[1], b[1]), min(union[2], b[2]), min(union[3], b[3]))
                )
            except (MissingCRS, CRSError) as e:
                log.info(f"Failed to compute crs for '{name}' ({store_key}): {e}")
                pass

        # static features: transform to input_crs as needed
        for name, path in static_paths.items():
            try:
                with rasterio.open(path) as src:
                    sb, scrs = src.bounds, src.crs
                if scrs is None:
                    continue
                b = sb if scrs == input_crs else transform_bounds(scrs, input_crs, *sb)
                bounds_per_feature[name] = b
                union = (
                    b
                    if union is None
                    else (max(union[0], b[0]), max(union[1], b[1]), min(union[2], b[2]), min(union[3], b[3]))
                )
            except FileNotFoundError as e:
                log.info(f"{path} was not found as {e}")
                pass

        out["bounds_per_feature"] = bounds_per_feature
        out["union_bounds"] = union

    return out
