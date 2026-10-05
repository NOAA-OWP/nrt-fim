from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rioxarray
import torch
import torch.nn.functional as F
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from trainer import FModel
from trainer.datasets.registry import VAR_NAME_MAP, compute_in_channels
from trainer.utils.normalization_methods import normalize_min_max

log = logging.getLogger(__name__)


# ----------------- checkpoint loading helpers -----------------


def _strip_prefix(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _pick_checkpoint(run_dir: Path, requested: str | None) -> Path:
    """
    Pick the checkpoint for a run

    :param run_dir: the main path of the training instance
    :param requested: the path of the torch model, if None, the last from run_dir will be picked
    :return: the checkpoint path
    """
    if requested:
        p = Path(requested)
        return p if p.is_absolute() else (run_dir / requested)
    # choose "best/last" or latest by mtime under saved_models/
    sm = run_dir / "saved_models"
    candidates = sorted(sm.glob("*.pt")) + sorted(sm.glob("*.pth")) + sorted(sm.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints in {sm}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model_from_checkpoint(cfg, ckpt_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """
    Create FModel from cfg and load weights from a variety of common checkpoint layouts.

    :param cfg: training config file that the checkpoint hase been made from
    :param ckpt_path: the path of the trained torch model
    :param device: device
    :return: NN model that has been loaded
    """
    ckpt_path = str(ckpt_path)
    in_channels = compute_in_channels(cfg)
    nn = FModel(
        num_classes=int(cfg.model.num_classes),
        in_channels=int(in_channels),
        device=device,
    ).to(device)

    obj = torch.load(ckpt_path, map_location=device)

    candidates = []
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "mlp", "model"):
            if k in obj and isinstance(obj[k], dict):
                candidates.append(obj[k])
    if not candidates and isinstance(obj, dict):
        # maybe it's already a state_dict-like mapping
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"Could not find state_dict in checkpoint: {ckpt_path}")

    state = candidates[0]

    # Handle prefixes like 'model.' or 'nn.'
    for pref in ("model.", "nn.", "module."):
        state = _strip_prefix(state, pref)

    missing, unexpected = nn.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning(f"Loaded with missing={list(missing)} unexpected={list(unexpected)}")

    nn.eval()
    return nn


#
# @torch.no_grad()
# def forward_full_image_tiled(
#     model: torch.nn.Module,
#     ds,                     # FloodDataset (already built with cfg; typically split="eval")
#     idx: int,               # which scene
#     tile_h: int,            # tile height
#     tile_w: int,            # tile width
#     device: str = "cpu",    # "cuda", "mps", or "cpu"
#     use_amp: bool = False,  # enable autocast if available for the device
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Tiled forward run over a full MODIS scene.
#
#     - Uses ds.features_* and ds._dyn_vars (dynamic) + statics via ds._open_static_da
#     - Pads tiles only to /32 for eval
#     - Works on CUDA, MPS (Apple), or CPU; autocast when available if use_amp=True
#
#     Returns:
#         pred_full (H, W)  : model output (still normalized; denorm later)
#         tgt_full  (H, W)  : normalized target (useful for metrics)
#     """
#
#     # ---------------- autocast helper (CUDA/MPS/CPU) ----------------
#     class _NoCtx:
#         def __enter__(self): return None
#         def __exit__(self, *a): return False
#
#     def _autocast_ctx(device_str: str, enabled: bool):
#         if not enabled:
#             return _NoCtx()
#         d = str(device_str).lower()
#         if "cuda" in d:
#             dev_type, dtype = "cuda", torch.float16
#         elif "mps" in d:
#             dev_type, dtype = "mps", torch.float16  # MPS supports fp16; autocast may be limited on older PyTorch
#         else:
#             # CPU autocast exists; bfloat16 is typical on CPU
#             dev_type, dtype = "cpu", torch.bfloat16
#         try:
#             return torch.autocast(device_type=dev_type, dtype=dtype)
#         except Exception:
#             # if autocast for this device/dtype isn't supported, silently no-op
#             return _NoCtx()
#
#     # ---------------- scene & target ----------------
#     r = ds.flood_instances.iloc[idx]
#     tif_path = r["tif_path"]
#     end_time = None
#     if "end_time" in ds.flood_instances.columns and r["end_time"] is not None:
#         end_time = np.datetime64(r["end_time"])
#
#     sat = rioxarray.open_rasterio(tif_path)
#     sat_water = sat[0]
#     perm_water = sat[4]
#     flood = sat_water - perm_water
#     flood = flood.where(flood < 0, 0)
#
#     H, W = int(sat.sizes["y"]), int(sat.sizes["x"])
#     pred_full = np.zeros((H, W), dtype=np.float32)
#     tgt_full  = np.zeros((H, W), dtype=np.float32)
#
#     # ---------------- temporal windows (if needed) ----------------
#     rho = int(ds.cfg.train.rho)
#     t_hourly = rho
#     t_3hr    = rho // 3
#
#     start_hourly_idx = end_hourly_idx = None
#     start_3hr_idx = end_3hr_idx = None
#
#     if ds.features_hourly and (ds.hourly_time is not None) and (end_time is not None):
#         end_hourly_idx = int(np.argmin(np.abs(ds.hourly_time - end_time)))
#         start_hourly_idx = end_hourly_idx - t_hourly
#
#     if ds.features_threeh and (ds.three_hourly_time is not None) and (end_time is not None):
#         end_3hr_idx = int(np.argmin(np.abs(ds.three_hourly_time - end_time)))
#         start_3hr_idx = end_3hr_idx - t_3hr
#
#     model.eval()
#
#
#     # ---------------- tiling ----------------
#     for y0 in range(0, H, tile_h):
#         for x0 in range(0, W, tile_w):
#             y1 = min(y0 + tile_h, H)
#             x1 = min(x0 + tile_w, W)
#
#             # target tile
#             flood_patch = flood.isel(y=slice(y0, y1), x=slice(x0, x1))
#             flood_patch_norm = normalize_min_max(flood_patch, ds.target_stats["band_1"], fix_nan_with="zero")
#
#             inputs_vars = []
#
#             # hourly dynamics (selected)
#             for name in ds.features_hourly:
#                 da = ds._dyn_vars[name]
#                 patch = ds._extract_patch_time_and_space(da, flood_patch_norm, start_hourly_idx, end_hourly_idx)
#                 stats_key = VAR_NAME_MAP[name].get("stats_key") or da.name
#                 patch = normalize_min_max(patch, ds.input_stats[stats_key], fix_nan_with="mean")
#                 inputs_vars.append(patch)  # (t, h, w)
#
#             # 3-hourly dynamics (selected)
#             for name in ds.features_threeh:
#                 da = ds._dyn_vars[name]
#                 patch = ds._extract_patch_time_and_space(da, flood_patch_norm, start_3hr_idx, end_3hr_idx)
#                 stats_key = VAR_NAME_MAP[name].get("stats_key") or da.name
#                 patch = normalize_min_max(patch, ds.input_stats[stats_key], fix_nan_with="mean")
#                 inputs_vars.append(patch)  # (t, h, w)
#
#             # statics (selected; open & align per tile)
#             for name in ds.features_static:
#                 store_key = VAR_NAME_MAP[name]["store"]
#                 path = ds._resolve_cfg_path(ds.cfg, store_key)
#                 if not path:
#                     continue
#                 sda = ds._open_static_da(path, name=name)
#                 spatch = ds._extract_static_patch_space(sda, flood_patch_norm)  # (h, w)
#                 stats_key = VAR_NAME_MAP[name].get("stats_key", name)
#                 if stats_key in ds.input_stats:
#                     spatch = normalize_min_max(spatch, ds.input_stats[stats_key], fix_nan_with="mean")
#                 else:
#                     spatch = np.nan_to_num(spatch, nan=0.0)
#                 inputs_vars.append(spatch[np.newaxis, ...])  # (1, h, w)
#
#             if not inputs_vars:
#                 pred_full[y0:y1, x0:x1] = 0.0
#                 tgt_full [y0:y1, x0:x1] = flood_patch_norm.values
#                 continue
#
#             # tensors + padding
#             inp = torch.tensor(np.concatenate(inputs_vars, axis=0), dtype=torch.float32)         # (C,h,w)
#             tgt = torch.tensor(flood_patch_norm.values[np.newaxis, ...], dtype=torch.float32)    # (1,h,w)
#
#             # pad to /32 only
#             inp_pad, tgt_pad, (ph, pw) = pad_for_model(inp, tgt, multiple=32, min_size=None)
#
#             bxin = inp_pad.unsqueeze(0).to(device, non_blocking=True)  # (1,C,H32,W32)
#
#             # forward with autocast if available for the device
#             with _autocast_ctx(device, use_amp):
#                 pred_pad = model(bxin)  # (1,1,H32,W32)
#
#             # unpad back to the original tile shape
#             pred_tile = unpad_to(pred_pad.squeeze(0).squeeze(0), ph, pw).cpu().numpy()  # (ph,pw)
#             tgt_tile = unpad_to(tgt_pad.squeeze(0).squeeze(0), ph, pw).cpu().numpy()  # (ph,pw)
#
#             pred_full[y0:y1, x0:x1] = pred_tile
#             tgt_full [y0:y1, x0:x1] = tgt_tile
#
#     return pred_full, tgt_full


def pad_for_model(
    x: torch.Tensor,  # (C,H,W) or (N,C,H,W) – handles both
    y: torch.Tensor | None = None,  # optional target, (1,H,W) or (N,1,H,W)
    multiple: int = 32,
    min_size: tuple[int, int] | None = None,  # (min_h, min_w). If None, skip this step.
):
    """
    Pads spatial dims to (at least) `min_size` and then to a multiple of `multiple`.

    Returns
    -------
        x_pad, y_pad, (orig_h, orig_w)
    """
    assert x.ndim in (3, 4), "x must be (C,H,W) or (N,C,H,W)"
    batched = x.ndim == 4

    if not batched:
        x = x.unsqueeze(0)  # (1,C,H,W)
        if y is not None and y.ndim == 3:
            y = y.unsqueeze(0)  # (1,1,H,W)

    _, _, H, W = x.shape
    orig_h, orig_w = H, W

    # 1) pad up to min_size (training only)
    if min_size is not None:
        min_h, min_w = int(min_size[0]), int(min_size[1])
        pad_h1 = max(0, min_h - H)
        pad_w1 = max(0, min_w - W)
        if pad_h1 or pad_w1:
            x = F.pad(x, (0, pad_w1, 0, pad_h1))
            if y is not None:
                y = F.pad(y, (0, pad_w1, 0, pad_h1))
            H += pad_h1
            W += pad_w1

    # 2) pad to nearest multiple
    pad_h2 = (multiple - (H % multiple)) % multiple
    pad_w2 = (multiple - (W % multiple)) % multiple
    if pad_h2 or pad_w2:
        x = F.pad(x, (0, pad_w2, 0, pad_h2))
        if y is not None:
            y = F.pad(y, (0, pad_w2, 0, pad_h2))

    if not batched:
        x = x.squeeze(0)
        if y is not None:
            y = y.squeeze(0)

    return x, y, (orig_h, orig_w)


def unpad_to(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """
    Crop back to (out_h, out_w) on the last two dims.

    Works for (H,W), (C,H,W) or (N,C,H,W).
    """
    if x.ndim == 2:
        return x[:out_h, :out_w]
    elif x.ndim == 3:
        return x[..., :out_h, :out_w]
    elif x.ndim == 4:
        return x[..., :out_h, :out_w]
    else:
        raise ValueError("Unsupported tensor rank for unpad_to.")


@torch.no_grad()
def forward_full_image_tiled(
    model: torch.nn.Module,
    ds,  # FloodDataset already built with cfg (typically split="eval")
    end_time: str | np.datetime64,  #
    tile_h: int = 1024,  # tile height in master-grid pixels
    tile_w: int = 1024,  # tile width  in master-grid pixels
    device: str = "cpu",  # "cuda", "mps", or "cpu"
    use_amp: bool = False,  # autocast when available
    aoi_bbox: tuple[float, float, float, float] | None = None,  # (minx,miny,maxx,maxy) in master-grid CRS
    aoi_bbox_crs: str = None,
    # override; else taken from ds.flood_instances[idx]["end_time"]
    master_path: str | None = None,  # override; else ds.master_path
):
    """
    Tiled forward run over an AOI defined in the master-grid CRS.

    - Does NOT read any target data.
    - AOI is a bbox in the master-grid CRS; if None, uses the full master grid.
    - Temporal indices are derived from `end_time` found in ds.flood_instances[idx] (eval split) or passed explicitly.
    - Dynamic features are reprojected/clipped to each master-grid tile; statics likewise.
    - Pads only to /32 to match the model’s downsampling, then unpads per tile.

    Returns
    -------
        pred_full : np.ndarray (H, W) in [0,1] (still normalized if your model outputs probs)
        transform : rasterio.Affine for the returned array
        crs       : rasterio.crs.CRS for the returned array
    """

    # ---------------- small autocast helper (CUDA/MPS/CPU) ----------------
    class _NoCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    def _autocast_ctx(device_str: str, enabled: bool):
        if not enabled:
            return _NoCtx()
        d = str(device_str).lower()
        if "cuda" in d:
            dev_type, dtype = "cuda", torch.float16
        elif "mps" in d:
            dev_type, dtype = "mps", torch.float16
        else:
            dev_type, dtype = "cpu", torch.bfloat16
        try:
            return torch.autocast(device_type=dev_type, dtype=dtype)
        except (RuntimeError, AttributeError, NotImplementedError, TypeError, ValueError) as e:
            log.debug(f"autocast unavailable for device={dev_type}: {e}")
            return _NoCtx()

    # ---------------- resolve master grid & AOI window ----------------
    mpath = str(master_path or ds.master_path)
    mgrid = rioxarray.open_rasterio(mpath)  # dims typically (band, y, x)
    mgrid = mgrid.squeeze(drop=True)  # drop band if present -> (y, x)
    m_crs = mgrid.rio.crs

    # Clip master grid to AOI bbox (still in master CRS) or use full
    if aoi_bbox is not None:
        if aoi_bbox_crs is not None:
            aoi_bbox = transform_bounds(
                aoi_bbox_crs, m_crs, aoi_bbox[0], aoi_bbox[1], aoi_bbox[2], aoi_bbox[3]
            )
        minx, miny, maxx, maxy = aoi_bbox
        mwin = mgrid.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=m_crs)
    else:
        mwin = mgrid

    H, W = int(mwin.sizes["y"]), int(mwin.sizes["x"])
    pred_full = np.zeros((H, W), dtype=np.float32)

    # preserve georef for output
    out_crs: CRS = m_crs
    out_transform: Affine = mwin.rio.transform()

    # ---------------- temporal windows (derived only if needed) ----------------
    # priority: explicit end_time arg -> ds.flood_instances[idx]["end_time"] if present
    if type(end_time) is str:
        end_time = np.datetime64(end_time)

    rho = int(ds.cfg.train.rho)
    t_hourly = rho
    t_3hr = rho // 3

    start_hourly_idx = end_hourly_idx = None
    start_3hr_idx = end_3hr_idx = None

    if ds.features_hourly and (ds.hourly_time is not None):
        end_hourly_idx = int(np.argmin(np.abs(ds.hourly_time - end_time)))
        start_hourly_idx = end_hourly_idx - t_hourly

    if ds.features_threeh and (ds.three_hourly_time is not None):
        end_3hr_idx = int(np.argmin(np.abs(ds.three_hourly_time - end_time)))
        start_3hr_idx = end_3hr_idx - t_3hr

    model.eval()

    # ---------------- tiling over the master window ----------------
    for y0 in range(0, H, tile_h):
        for x0 in range(0, W, tile_w):
            y1 = min(y0 + tile_h, H)
            x1 = min(x0 + tile_w, W)

            # make a master-grid tile (acts as spatial "reference patch")
            master_tile: xr.DataArray = mwin.isel(y=slice(y0, y1), x=slice(x0, x1))

            inputs_vars = []

            # ---- hourly dynamics (selected) ----
            if ds.features_hourly:
                if start_hourly_idx is None or end_hourly_idx is None:
                    raise ValueError(
                        "Hourly features requested but hourly time index could not be determined."
                    )
                for name in ds.features_hourly:
                    da = ds.inputs_dict["dyn_vars"][name]  # xr.DataArray (time,y,x) on NWM grid
                    patch = ds._extract_patch_time_and_space(  # -> np.ndarray (t, h, w) on master grid
                        da, master_tile, start_hourly_idx, end_hourly_idx
                    )
                    stats_key = VAR_NAME_MAP[name].get("stats_key", None)
                    patch = normalize_min_max(patch, ds.input_stats[stats_key], fix_nan_with="mean")
                    inputs_vars.append(patch)

            # ---- three-hourly dynamics (selected) ----
            if ds.features_threeh:
                if start_3hr_idx is None or end_3hr_idx is None:
                    raise ValueError(
                        "Three-hourly features requested but 3-hour time index could not be determined."
                    )
                for name in ds.features_threeh:
                    da = ds.inputs_dict["dyn_vars"][name]
                    patch = ds._extract_patch_time_and_space(da, master_tile, start_3hr_idx, end_3hr_idx)
                    stats_key = VAR_NAME_MAP[name].get("stats_key", None)
                    if stats_key in ds.input_stats:
                        patch = normalize_min_max(patch, ds.input_stats[stats_key], fix_nan_with="mean")
                    inputs_vars.append(patch)

            # ---- statics (selected) ----
            for name in ds.features_static:
                store_key = VAR_NAME_MAP[name]["store"]
                path = ds._resolve_cfg_path(ds.cfg, store_key)
                if not path:
                    continue
                sda = ds._open_static_da(path, name=name)  # xr.DataArray (y,x) with its own CRS
                spatch = ds._extract_static_patch_space(
                    sda, master_tile
                )  # -> np.ndarray (h,w) on master grid
                stats_key = VAR_NAME_MAP[name].get("stats_key", None)
                if stats_key in ds.input_stats:
                    spatch = normalize_min_max(spatch, ds.input_stats[stats_key], fix_nan_with="mean")
                else:
                    spatch = np.nan_to_num(spatch, nan=0.0)
                inputs_vars.append(spatch[np.newaxis, ...])  # (1,h,w)

            if not inputs_vars:
                # Nothing to feed → leave zeros (or raise); here we keep zeros
                continue

            # ----- tensorize + pad (to /32 only) -----
            inp = torch.tensor(np.concatenate(inputs_vars, axis=0), dtype=torch.float32)  # (C,h,w)
            inp_pad, _, (ph, pw) = pad_for_model(inp, y=None, multiple=32, min_size=None)
            bxin = inp_pad.unsqueeze(0).to(device, non_blocking=True)  # (1,C,H32,W32)

            # ----- forward -----
            with _autocast_ctx(device, use_amp):
                pred_pad = model(bxin)  # (1,1,H32,W32)

            # ----- unpad back to tile -----
            pred_tile = unpad_to(pred_pad.squeeze(0).squeeze(0), ph, pw).detach().cpu().numpy()

            # ----- write into mosaic -----
            pred_full[y0:y1, x0:x1] = pred_tile

    return pred_full, out_transform, out_crs
