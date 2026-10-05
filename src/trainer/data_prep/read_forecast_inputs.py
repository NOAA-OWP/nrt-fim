from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import s3fs
import xarray as xr
from omegaconf import DictConfig
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from trainer.datasets.registry import VAR_NAME_MAP
from trainer.utils.geo_utils import _resolve_dynamic_da, _to_crs_obj, infer_crs
from trainer.utils.utils import resolve_cfg_path

log = logging.getLogger(__name__)


def read_selected_forecast_inputs(
    cfg: DictConfig,
    compute_bounds: bool = False,
    var_name_map: dict[str, dict[str, Any]] = VAR_NAME_MAP,
) -> dict[str, Any]:
    """
    Read forecast data from noaa-nwm-pds.s3 bucket.

    The function works similar to the identical one in read_input.py for retrospective data
    """
    feats_hourly = sorted([f for f in (cfg.features.get("hourly") or []) if f in var_name_map])
    feats_threeh = sorted([f for f in (cfg.features.get("three_hourly") or []) if f in var_name_map])
    feats_static = sorted([f for f in (cfg.features.get("static") or []) if f in var_name_map])

    dyn_ds: dict[str, xr.Dataset] = {}
    crs_by_store: dict[str, CRS] = {}
    dyn_vars: dict[str, xr.DataArray] = {}
    hourly_time = None
    three_hourly_time = None

    ## hourly
    needed = []
    var_list = []
    if len(feats_hourly) > 0:
        for name in feats_hourly:
            meta = var_name_map[name]
            var_list.append(meta["var"])
        component = resolve_cfg_path(cfg, meta["store"])
        product = str(Path(component).parent)
        needed.append((component, var_list, product))
    # needed = sorted(set(needed))

    # finding the starting time:
    _, _, _, base_dt = _parse_iso_ymd(cfg["forecast_end_time"])
    start_dt = base_dt - timedelta(hours=cfg["train"]["rho"] + 3)
    start_date_ymd = start_dt.strftime("%Y%m%d")

    if needed:
        ds = nwm_timeseries(
            date=start_date_ymd,  # ISO ok
            component=component,
            var_list=var_list,
            product=product,
            leads=range(0, cfg["train"]["rho"] + 3),
            spatial_chunks=("auto", "auto"),
        )
        ## because all forcings are here
        for name in feats_hourly:
            meta = var_name_map[name]
            store_key = meta["store"]
            dyn_ds[store_key] = ds
            da = _resolve_dynamic_da(name, dyn_ds, var_name_map=var_name_map)
            dyn_vars[name] = da

            crs = ds.rio.crs
            crs_by_store[name] = crs

    ## three-hourly --> Terrain_router
    if "terrain_router" in feats_threeh:
        needed = []
        name = "terrain_router"
        meta = var_name_map[name]
        var_list = [meta["var"]]
        component = resolve_cfg_path(cfg, meta["store"])
        product = str(Path(component).parent)
        needed.append((component, var_list, product))
        if needed:
            # for component, var_list, product in needed:
            ds = nwm_timeseries(
                date=start_date_ymd,  # ISO ok
                component=component,
                var_list=var_list,
                product=product,
                leads=range(0, cfg["train"]["rho"] + 6),
                spatial_chunks=("auto", "auto"),
            )
            store_key = meta["store"]
            dyn_ds[store_key] = ds
            da = _resolve_dynamic_da(name, dyn_ds, var_name_map=var_name_map)
            dyn_vars[name] = da

            crs = ds.rio.crs
            crs_by_store[name] = crs

    ### three-hourly --> other variables
    needed = []
    var_list = []
    for name in feats_threeh:
        if name != "terrain_router":
            meta = var_name_map[name]
            var_list.append(meta["var"])
            component = resolve_cfg_path(cfg, meta["store"])
            product = str(Path(component).parent)
            needed.append((component, var_list, product))
    if needed:
        # for component, var_list, product in needed:
        ds = nwm_timeseries(
            date=start_date_ymd,  # ISO ok
            component=component,
            var_list=var_list,
            product=product,
            leads=range(0, cfg["train"]["rho"] + 6),
            spatial_chunks=("auto", "auto"),
        )
        for name in feats_threeh:
            if name != "terrain_router":
                meta = var_name_map[name]
                store_key = meta["store"]
                dyn_ds[store_key] = ds
                da = _resolve_dynamic_da(name, dyn_ds, var_name_map=var_name_map)
                dyn_vars[name] = da

                crs = ds.rio.crs
                crs_by_store[name] = crs

        # materialize named variables from dyn_ds and attach CRS
    for name in feats_hourly + feats_threeh:
        da = dyn_vars[name]
        if ("time" in da.dims) and (name in feats_hourly) and (hourly_time is None):
            hourly_time = da["time"].values
        if ("time" in da.dims) and (name in feats_threeh) and (three_hourly_time is None):
            three_hourly_time = da["time"].values

    input_crs = list(crs_by_store.values())[0]
    # else:
    #     input_crs = CRS.from_epsg(4326)

    # static paths (lazy—don't open unless asked for bounds)
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

    if compute_bounds:
        bounds_per_feature: dict[str, tuple[float, float, float, float]] = {}
        union: tuple[float, float, float, float] | None = None

        # dynamic: prefer coordinate min/max if x/y present
        for name, da in dyn_vars.items():
            try:
                if "x" in da.coords and "y" in da.coords:
                    b = (float(da.x.min()), float(da.y.min()), float(da.x.max()), float(da.y.max()))
                else:
                    # fallback to rioxarray bounds if transform is set
                    b = da.rio.bounds()
                # transform to input_crs if store's CRS differs
                store_key = f"{var_name_map[name].get('product', 'medium_range_blend')}:{var_name_map[name]['component']}"
                da_crs = crs_by_store.get(store_key, input_crs)
                if da_crs and input_crs and da_crs != input_crs:
                    b = transform_bounds(da_crs, input_crs, *b, densify_pts=0)
                bounds_per_feature[name] = b
                union = (
                    b
                    if union is None
                    else (max(union[0], b[0]), max(union[1], b[1]), min(union[2], b[2]), min(union[3], b[3]))
                )
            except (PermissionError, ValueError):
                pass

        # static rasters
        for name, path in static_paths.items():
            try:
                with rasterio.open(path) as src:
                    sb, scrs = src.bounds, src.crs
                if scrs is None:
                    continue
                b = sb if scrs == input_crs else transform_bounds(scrs, input_crs, *sb, densify_pts=0)
                bounds_per_feature[name] = b
                union = (
                    b
                    if union is None
                    else (max(union[0], b[0]), max(union[1], b[1]), min(union[2], b[2]), min(union[3], b[3]))
                )
            except FileNotFoundError as e:
                log.warning(f"static path is not available {path}: {e}")
                pass

        out["bounds_per_feature"] = bounds_per_feature
        out["union_bounds"] = union

    return out


def _parse_iso_ymd(date_str: str) -> tuple[int, int, int, datetime]:
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    base_dt = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    return base_dt.year, base_dt.month, base_dt.day, base_dt


def _key_for(date_ymd: str, product: str, cycle: str, component: str, lead: int) -> str:
    # nwm.YYYYMMDD/<product>/nwm.<cycle>.<product>.<component>.fXXX.conus.nc
    file_name = Path(component).name.split(".conus")[0][:-3]
    path = f"nwm.{date_ymd}/{product}/{file_name}{lead:03d}.conus.nc"
    return path
    # return f"nwm.{date_ymd}/{product}/nwm.{cycle}.{product}.{component}.f{lead:03d}.conus.nc"


def nwm_timeseries(
    date: str,  # "YYYYMMDD" or ISO "YYYY-MM-DDTHH:MM:SS[Z]"
    component: str,  # "channel_rt", "terrain_rt", "land", ...
    var_list: list,  # variable in file
    leads: Iterable[int] = range(0, 73),
    bucket: str = "noaa-nwm-pds",
    product: str = "medium_range_blend",
    cycle: str = "t00z",
    spatial_chunks: tuple[int | str, int | str] = ("auto", "auto"),
) -> xr.Dataset:
    """
    Extract the timeseries data

    :param date: the starting time
    :param component: the file_name comonent
    :param var_list: all variables in a ds that we need to collect. all forcings are in a single file, etc.
    :param leads: lead time to gather data
    :param bucket: the bucket name
    :param product: short or medium range. now it is hard coded. need to be taken care of in the future
    :param cycle: the cycle time. it is mostly 00. but can be different
    :param spatial_chunks: since it netcdf file, mostly "auto"
    :return: ds
    """
    ## TODO: bucket name, product, and cycle should be modified in the future.
    # It also always starts at time 00:00:00, which should change in the future

    y, m, d, base_dt = _parse_iso_ymd(date)
    date_ymd = f"{y:04d}{m:02d}{d:02d}"

    fs = s3fs.S3FileSystem(anon=True)
    existing: list[tuple[int, str]] = []
    for lead in leads:
        key = _key_for(date_ymd, product, cycle, component, lead)
        if fs.exists(f"{bucket}/{key}"):
            existing.append((lead, key))
    if not existing:
        raise FileNotFoundError(
            f"No files in s3://{bucket}/nwm.{date_ymd}/{product}/ for component={component!r}"
        )

    def open_one(lead: int, key: str) -> xr.Dataset:
        url = f"s3://{bucket}/{key}"
        ds = xr.open_dataset(
            url,
            engine="h5netcdf",
            backend_kwargs={"storage_options": {"anon": True}},
            chunks={"time": 1, "y": spatial_chunks[0], "x": spatial_chunks[1]},
        )
        # if var_list not in ds.variables:
        #     raise KeyError(f"{var!r} not found in {url}. Available: {list(ds.data_vars)}")
        crs_obj = _to_crs_obj(infer_crs(ds))
        ds = ds[var_list]
        # writing crs
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        ds = ds.rio.write_crs(crs_obj, inplace=False)

        # ensure time dim, stamp forecast valid time
        if "time" not in ds.dims:
            singleton = next((d for d in ds.dims if ds.dims[d] == 1), None)
            if singleton is None:
                raise ValueError(f"No singleton time-like dim in {url}")
            if singleton != "time":
                ds = ds.rename({singleton: "time"})
        t = base_dt + timedelta(hours=int(lead))
        # ds = ds.assign_coords(time=[np.datetime64(t.isoformat())])
        ds = ds.assign_coords(time=[np.datetime64(int(t.astimezone(UTC).timestamp()), "s")])
        return ds

    parts = [open_one(lead, key) for lead, key in existing]
    out = xr.concat(
        parts,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    ).sortby("time")

    return out
