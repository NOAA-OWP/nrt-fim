from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rioxarray
import torch
import torch.nn.functional as F
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rioxarray.exceptions import MissingCRS, NoDataInBounds
from torch.utils.data import Dataset

from trainer.data_prep.read_forecast_inputs import read_selected_forecast_inputs
from trainer.data_prep.read_inputs import read_selected_inputs

# from trainer.utils.utils import create_master_from_da  # your existing util
from trainer.datasets.registry import VAR_NAME_MAP
from trainer.utils.geo_utils import _ensure_master_grid
from trainer.utils.normalization_methods import normalize_min_max

log = logging.getLogger(__name__)


class FloodDataset(Dataset):
    """
    Lightweight map-style dataset:

      - expects index CSV and split lists (train/val/test)
      - loads precomputed stats JSONs
      - builds inputs by slicing + reprojecting on the fly (can be moved offline later)
    """

    def __init__(self, cfg: DictConfig, split: Literal["train", "eval", "user_defined"] = "train"):
        self.cfg = cfg
        self.split = split

        self.full_image_eval = False  # this is for evaluation part to forward run with full image size

        # index + split
        if self.split in ["train", "eval"]:
            df = pd.read_csv(cfg.data_sources.index_csv)
            flood_img_paths = Path(cfg.data_sources.splits_dir, f"{self.split}.json")
            with open(flood_img_paths) as f:
                flood_ids = json.load(f)
            use_ids = sorted(int(s) for s in flood_ids if s.strip())
            self.flood_instances = df[df.flood_id.isin(use_ids)].reset_index(drop=True)
        # else:   # == user_define
        #     self.flood_instances =

        # ---------------- interpreting requested features to check if they are available ----------------
        self._interpret_requested_features()

        # ---------------- read/prepare inputs (ALL input reading consolidated here) ----------------
        if cfg["forecast_data_source"]:
            self.inputs_dict = read_selected_forecast_inputs(cfg)
        else:
            self.inputs_dict = read_selected_inputs(cfg)

        # ---------------- read master_grid path ---------------------------------------------------
        self.master_path = _ensure_master_grid(self.cfg, self.inputs_dict)

        # Capture hourly time and three-hourly time
        if len(self.features_hourly) > 0:
            self.hourly_time = np.asarray(self.inputs_dict["dyn_vars"][self.features_hourly[0]]["time"])
        if len(self.features_threeh) > 0:
            self.three_hourly_time = np.asarray(self.inputs_dict["dyn_vars"][self.features_threeh[0]]["time"])

        # ---------------- read/prepare stats ----------------
        input_file_name = (
            "input_stats" + "_" + str(cfg.train.start_time) + "_" + str(cfg.train.end_time) + ".json"
        )
        input_stats_path = Path(self.cfg.project_root) / "statistics" / input_file_name
        with open(input_stats_path) as f:
            self.input_stats = json.load(f)

        target_file_name = (
            "target_stats" + "_" + str(cfg.train.start_time) + "_" + str(cfg.train.end_time) + ".json"
        )
        target_stats_path = Path(self.cfg.project_root) / "statistics" / target_file_name
        with open(target_stats_path) as f:
            self.target_stats = json.load(f)

    def __len__(self):
        return len(self.flood_instances)

    def __getitem__(self, i: int):
        r = self.flood_instances.iloc[i]
        tif_path = r["tif_path"]
        end_time = np.datetime64(r["end_time"]) if pd.notnull(r["end_time"]) else None

        # ── build target (your current convention uses band 0 only) ──:wq!
        sat = rioxarray.open_rasterio(tif_path)
        sat_water = sat[0]
        flood0 = sat_water
        flood = flood0.where(flood0 > 0, 0)  # a xarray.where function. different from np.where

        # ── sample spatial window ──
        flood_patch = self.spatial_sampling(
            image=flood,
            full_image_eval=self.full_image_eval,
        )

        # normalize target
        flood_patch_norm = normalize_min_max(
            flood_patch,
            self.target_stats["band_1"],
            fix_nan_with=self.cfg["normalization"]["fix_nan_with_obs"],
        )

        # ── temporal windows per group ──
        rho = int(self.cfg.train.rho)
        t_hourly = rho
        t_3hr = rho // 3

        start_hourly_idx = end_hourly_idx = None
        if (len(self.features_hourly) > 0) and (self.hourly_time is not None) and (end_time is not None):
            end_hourly_idx = int(np.argmin(np.abs(self.hourly_time - end_time)))
            start_hourly_idx = end_hourly_idx - t_hourly

        start_3hr_idx = end_3hr_idx = None
        if (
            (len(self.features_threeh) > 0)
            and (self.three_hourly_time is not None)
            and (end_time is not None)
        ):
            end_3hr_idx = int(np.argmin(np.abs(self.three_hourly_time - end_time)))
            start_3hr_idx = end_3hr_idx - t_3hr

        inputs_vars: list[np.ndarray] = []

        # ── hourly dynamics ──
        if start_hourly_idx is not None and end_hourly_idx is not None:
            for name in self.features_hourly:
                da = self.inputs_dict["dyn_vars"][name]
                patch = self._extract_patch_time_and_space(
                    da, flood_patch_norm, start_hourly_idx, end_hourly_idx
                )
                stats_key = VAR_NAME_MAP[name].get("stats_key") or da.name
                if stats_key in self.input_stats:
                    patch = normalize_min_max(
                        patch,
                        self.input_stats[stats_key],
                        fix_nan_with=self.cfg["normalization"]["fix_nan_with_dyn_inp"],
                    )
                else:
                    patch = np.nan_to_num(patch, nan=0.0)
                inputs_vars.append(patch)
        elif self.features_hourly:
            log.debug("Skipping hourly features for this sample: end_time/indices unavailable.")

        # ── three-hourly dynamics ──
        if start_3hr_idx is not None and end_3hr_idx is not None:
            for name in self.features_threeh:
                da = self.inputs_dict["dyn_vars"][name]
                patch = self._extract_patch_time_and_space(da, flood_patch_norm, start_3hr_idx, end_3hr_idx)
                stats_key = VAR_NAME_MAP[name].get("stats_key") or da.name
                if stats_key in self.input_stats:
                    patch = normalize_min_max(
                        patch,
                        self.input_stats[stats_key],
                        fix_nan_with=self.cfg["normalization"]["fix_nan_with_dyn_inp"],
                    )
                else:
                    patch = np.nan_to_num(patch, nan=0.0)
                inputs_vars.append(patch)
        elif self.features_threeh:
            log.debug("Skipping three-hourly features for this sample: end_time/indices unavailable.")

        # ── statics (open per-sample) ──
        for name in self.features_static:
            # prefer pre-resolved path from inputs_dict; fall back to cfg resolver
            try:
                path = self.inputs_dict["static_paths"][name]
            except KeyError as e:
                raise KeyError("Static paths not available for this sample.") from e
            if not path:
                store_key = VAR_NAME_MAP[name]["store"]
                path = self._resolve_cfg_path(self.cfg, store_key)

            if not path:
                log.warning(f"[static] '{name}' has no path in cfg, skipping.")
                continue

            sda = self._open_static_da(path, name=name)
            spatch = self._extract_static_patch_space(sda, flood_patch_norm)  # (h, w)
            stats_key = VAR_NAME_MAP[name].get("stats_key", name)
            if stats_key in self.input_stats:
                spatch = normalize_min_max(
                    spatch,
                    self.input_stats[stats_key],
                    fix_nan_with=self.cfg["normalization"]["fix_nan_with_static_inp"],
                )
            else:
                spatch = np.nan_to_num(spatch, nan=0.0)
            inputs_vars.append(spatch[np.newaxis, ...])  # (1, h, w)

        if not inputs_vars:
            raise ValueError("No inputs produced: all requested feature groups were empty or skipped.")

        # tensors + padding
        input_tensor = torch.tensor(np.concatenate(inputs_vars, axis=0), dtype=torch.float32)  # (C, H, W)
        target_tensor = torch.tensor(flood_patch_norm.values[np.newaxis, ...], dtype=torch.float32)

        input_tensor, target_tensor = self._pad_to_multiple_32(input_tensor, target_tensor)
        return input_tensor, target_tensor

    def _extract_patch_time_and_space(
        self,
        da: xr.DataArray,
        master_patch: xr.DataArray,
        start_idx: int | None,
        end_idx: int | None,
    ) -> np.ndarray:
        """
        Extract a spatio‐temporal patch:

          - time slice from start_time_index to end_time_index inclusive
          - spatial window defined by (x0,y0,x_size,y_size), aligned to master_patch

        Returns
        -------
            matched.values: a numpy array of shape (time, y_size, x_size)
        """
        temp = da
        if start_idx is not None and end_idx is not None:
            temp = da.isel(time=slice(start_idx, end_idx))
        # temp = temp.rio.write_crs(self.input_crs, inplace=False)

        mp_crs = master_patch.rio.crs
        minx, miny, maxx, maxy = master_patch.rio.bounds()
        bx_minx, bx_miny, bx_maxx, bx_maxy = transform_bounds(
            mp_crs, self.inputs_dict["input_crs"], minx, miny, maxx, maxy
        )

        clipped = temp.rio.clip_box(
            minx=bx_minx, miny=bx_miny, maxx=bx_maxx, maxy=bx_maxy, crs=self.inputs_dict["input_crs"]
        )
        matched = clipped.rio.reproject_match(master_patch, resampling=Resampling.bilinear)
        return matched.values

    def _pad_to_multiple_32(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Padding to the extent of max x and y sizes defined in cfg

        :param input_tensor: inputs
        :param target_tensor:
        :return: padded input tensor and padded target tensor
        """
        want_h, want_w = int(self.cfg.train.No_pixels_y), int(self.cfg.train.No_pixels_x)
        _, H, W = input_tensor.shape

        # 1) pad up to requested patch size
        pad_h1 = max(0, want_h - H)
        pad_w1 = max(0, want_w - W)
        if pad_h1 or pad_w1:
            input_tensor = F.pad(input_tensor, (0, pad_w1, 0, pad_h1))
            target_tensor = F.pad(target_tensor, (0, pad_w1, 0, pad_h1))
            H += pad_h1
            W += pad_w1

        # 2) pad to /32
        pad_h2 = (32 - H % 32) % 32
        pad_w2 = (32 - W % 32) % 32
        if pad_h2 or pad_w2:
            input_tensor = F.pad(input_tensor, (0, pad_w2, 0, pad_h2))
            target_tensor = F.pad(target_tensor, (0, pad_w2, 0, pad_h2))
        # keep last target plane if needed
        return input_tensor, target_tensor[-1:, :, :]

    def _interpret_requested_features(self):
        feats = getattr(self.cfg, "features", None)
        self.features_hourly = sorted(getattr(feats, "hourly", []) or [])
        self.features_threeh = sorted(getattr(feats, "three_hourly", []) or [])
        self.features_static = sorted(getattr(feats, "static", []) or [])

        self.features_hourly = self._filter_known(self.features_hourly, "hourly")
        self.features_threeh = self._filter_known(self.features_threeh, "three_hourly")
        self.features_static = self._filter_known(self.features_static, "static")

        if not (self.features_hourly or self.features_threeh or self.features_static):
            raise ValueError("No features selected. Set cfg.features with at least one feature name.")

    # ────────────────────────────────────────────────────────────────
    # helpers
    # ────────────────────────────────────────────────────────────────
    def _filter_known(self, names: list[str], group: str) -> list[str]:
        if not names:
            return []
        known, unknown = [], []
        for n in names:
            meta = VAR_NAME_MAP.get(n)
            if meta is None or meta.get("group") != group:
                unknown.append(n)
            else:
                known.append(n)
        if unknown:
            log.warning(f"[features] Unknown/incorrect {group} features skipped: {unknown}")
        return known

    def _resolve_cfg_path(self, cfg: DictConfig, dotted: str) -> str | None:
        key = dotted[4:] if dotted.startswith("cfg.") else dotted
        return OmegaConf.select(cfg, key)

    def _resolve_dynamic_da(self, name: str) -> xr.DataArray:
        meta = VAR_NAME_MAP[name]
        ds = self._dyn_ds[meta["store"]]
        da = ds[meta["var"]]
        if meta.get("sel"):
            da = da.sel(**meta["sel"])
        # Give a stable name for stats lookup & debugging
        da = da.rename(meta.get("stats_key", meta["var"]))
        return da

    def _open_static_da(self, path: str | Path, name: str):
        da = rioxarray.open_rasterio(str(path))
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)
        if da.rio.crs is None:
            print("static inputs {name} does not have crs")
            exit()
        da.name = name
        return da

    def _extract_static_patch_space(self, arr2d: xr.DataArray, master_patch: xr.DataArray) -> np.ndarray:
        """
        Extract static patch space

        Safely align a static raster to the master_patch grid.
        - If there is spatial overlap: clip (fast) then reproject.
        - If not: fall back to reproject_match on the whole raster (yields all-nodata).
        Uses nearest for categorical layers like 'flow_direction'

        :param arr2d: the static input that needs to be patched.

        :param master_patch: the patch that the static input will be patched based on.
        :return: patched statics
        """
        name = arr2d.name or "static"
        resampling = Resampling.nearest if name in {"flow_direction"} else Resampling.bilinear

        # ensure CRS exists
        if arr2d.rio.crs is None:
            arr2d = arr2d.rio.write_crs("EPSG:4326", inplace=False)

        mp_crs = master_patch.rio.crs
        minx, miny, maxx, maxy = master_patch.rio.bounds()

        try:
            # transform master bounds into arr2d CRS
            bx_minx, bx_miny, bx_maxx, bx_maxy = transform_bounds(
                mp_crs, arr2d.rio.crs, minx, miny, maxx, maxy
            )
            ax_min, ay_min, ax_max, ay_max = arr2d.rio.bounds()

            # check intersection before clipping
            overlap = not (bx_maxx < ax_min or bx_minx > ax_max or bx_maxy < ay_min or bx_miny > ay_max)
            if overlap:
                clipped = arr2d.rio.clip_box(
                    minx=bx_minx, miny=bx_miny, maxx=bx_maxx, maxy=bx_maxy, crs=arr2d.rio.crs
                )
                matched = clipped.rio.reproject_match(master_patch, resampling=resampling)
                return matched.values
        except (NoDataInBounds, MissingCRS) as e:
            # fall back if transform/clip fails for any reason
            log.debug(f"[static] clip_box fallback ({name}): {e}")

        # fallback: reproject entire raster to the patch grid (will be nodata if no overlap)
        matched = arr2d.rio.reproject_match(master_patch, resampling=resampling)
        return matched.values

    def spatial_sampling(
        self,
        image: xr.DataArray,
        full_image_eval: bool = False,
    ):
        """
        Spatial sampling from a large image/tif xarray file

        :param image: the image that a sample will be taken from
        :param full_image_eval: True means the whole image will be taken as one single sample
        :return: patched image
        """
        H, W = int(image.sizes["y"]), int(image.sizes["x"])
        want_h, want_w = int(self.cfg.train.No_pixels_y), int(self.cfg.train.No_pixels_x)

        if full_image_eval:
            y0, x0 = 0, 0
            crop_h, crop_w = H, W
        else:
            crop_h = min(want_h, H)
            crop_w = min(want_w, W)
            max_y, max_x = H - crop_h, W - crop_w
            y0 = 0 if max_y <= 0 else int(np.random.randint(0, max_y + 1))
            x0 = 0 if max_x <= 0 else int(np.random.randint(0, max_x + 1))

        y1, x1 = y0 + crop_h, x0 + crop_w
        flood_patch = image.isel(y=slice(y0, y1), x=slice(x0, x1))
        return flood_patch
