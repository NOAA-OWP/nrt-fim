from __future__ import annotations
import numpy as np
import xarray as xr
from typing import Literal

def normalize_min_max(
        val: np.ndarray | xr.DataArray,
        stats: dict,
        fix_nan_with: Literal["mean", "zero", "min", "max"] = "mean",
) -> np.ndarray | xr.DataArray:
    """
    Min–max normalize an array (or xarray) to [0,1], then replace any NaNs via one of four strategies.

    Parameters
    ----------
    val : np.ndarray or xr.DataArray
        The raw data to normalize.
    stats : dict
        Dictionary with keys "min", "max", "mean" giving the original data stats.
    fix_nan_with : {"mean","zero","min","max"}
        How to fill any NaNs after normalization.
        - "mean"  → fill with normalized mean = (mean - min)/(max - min)
        - "zero"  → fill with 0.0
        - "min"   → same as zero (the minimum of the normalized range)
        - "max"   → fill with 1.0

    Returns
    -------
    normed : same type as `val`
        The normalized data, with NaNs replaced.
    """
    mn = stats["min"]
    mx = stats["max"]
    rng = mx - mn
    if rng == 0:
        raise ValueError(f"Cannot normalize when max==min=={mn}")
    # do the normalization
    normed = (val - mn) / rng

    # decide fill value in the normalized space

    if fix_nan_with == "mean":
        fill = (stats["mean"] - mn) / rng
    elif fix_nan_with in ("zero", "min"):
        fill = 0.0
    elif fix_nan_with == "max":
        fill = 1.0
    elif fix_nan_with is False or fix_nan_with is None:
        # Skip full sanitization (for observations). Still demote ±Inf -> NaN so masking works.
        if isinstance(normed, xr.DataArray):
            normed = normed.where(np.isfinite(normed))
        else:
            a = np.asarray(normed)
            normed = np.where(np.isfinite(a), a, np.nan).astype("float32", copy=False)
        return normed
    else:
        raise ValueError(f"Unknown fix_nan_with={fix_nan_with}")
    normed_treated = sanitize_numeric(
        normed,
        fill=fill,  # what you want use to fill the data with
        treat_255_as_nodata=True,  # sometimes 255 is a standard value ifor no data
        treat_1e20_as_nodata=True,  # sometimes 1e+20 is a standard value ifor no data
        also_flag_large_as_nodata=True,  # catches any huge numeric sentinels
        large_threshold=1e19,  # if also_flag_large_as_nodata==True, any number larger than this will be nodata
        extra_nodata_values=None,  # add [-9999, -999] etc if needed, None will skip this part
        out_dtype="float32",
    )

    return normed_treated

def sanitize_numeric(
    arr,
    fill=0.0,
    treat_255_as_nodata=True,
    treat_1e20_as_nodata=True,
    extra_nodata_values=None,   # e.g., [ -9999, 3.4028235e38 ]
    also_flag_large_as_nodata=False,  # treat huge magnitudes like 1e20 as nodata
    large_threshold=1e19,
    out_dtype="float32",
):
    """
    Replace NaN/Inf and common nodata sentinels with a safe fill value.

    - Treats ±Inf as nodata
    - Optionally treats 255 (typical for uint8 masks) as nodata
    - Optionally treats 1e20 as nodata (common float sentinel)
    - Can also flag any |x| >= large_threshold as nodata
    - Preserves xarray coords/attrs if input is a DataArray
    """
    def _sanitize_np(a: np.ndarray) -> np.ndarray:
        """sanitize numpy arrays. xarray format is written sanitize_numeric"""
        a = a.astype(np.float32, copy=True)
        mask = ~np.isfinite(a)
        if treat_1e20_as_nodata:
            mask |= np.isclose(a, 1.0e20)
        if also_flag_large_as_nodata:
            mask |= (np.abs(a) >= large_threshold)
        if treat_255_as_nodata:
            mask |= (a == 255)
        if extra_nodata_values:
            for v in extra_nodata_values:
                # choose equality for ints, isclose for floats
                if isinstance(v, (int, np.integer)):
                    mask |= (a == v)
                else:
                    mask |= np.isclose(a, v)
        a[mask] = fill
        return a.astype(out_dtype, copy=False)

    if isinstance(arr, xr.DataArray):
        a = arr.astype(np.float32)
        mask = ~np.isfinite(a)
        if treat_1e20_as_nodata:
            mask = mask | xr.apply_ufunc(np.isclose, a, 1.0e20)
        if also_flag_large_as_nodata:
            mask = mask | (np.abs(a) >= large_threshold)
        if treat_255_as_nodata:
            mask = mask | (a == 255)
        if extra_nodata_values:
            for v in extra_nodata_values:
                if isinstance(v, (int, np.integer)):
                    mask = mask | (a == v)
                else:
                    mask = mask | xr.apply_ufunc(np.isclose, a, v)
        cleaned = a.where(~mask, other=fill)
        return cleaned.astype(out_dtype)
    else:
        # assume numpy array-like
        return _sanitize_np(np.asarray(arr))

def denormalize_min_max(arr: np.ndarray, stats: dict) -> np.ndarray:
    """
    Reverse of min–max normalize_min_max: x = norm * (max - min) + min
    stats must contain keys: "min", "max".
    """
    mn = float(stats["min"])
    mx = float(stats["max"])
    scale = mx - mn
    if scale <= 0:
        # Degenerate case: avoid divide-by-zero behavior
        return np.full_like(arr, mn)
    out = arr * scale + mn
    return out