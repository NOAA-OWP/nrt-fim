from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS as PJCRS
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rioxarray.exceptions import MissingCRS
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

log = logging.getLogger(__name__)


def infer_crs(obj) -> CRS:
    """
    Return a rasterio.CRS for an xarray Dataset/DataArray.

    Prefers rioxarray's attached CRS; otherwise tries 'crs' variable WKT.
    Raises if nothing is found.
    """
    # 1) rioxarray accessor
    rio = getattr(obj, "rio", None)
    crs = getattr(rio, "crs", None)
    if crs is not None:
        return crs  # already a rasterio.CRS

    # 2) Dataset with 'crs' var carrying WKT
    if isinstance(obj, xr.Dataset) and "crs" in obj:
        crs_da = obj["crs"]
        wkt = crs_da.attrs.get("spatial_ref") or crs_da.attrs.get("crs_wkt")
        if wkt:
            return CRS.from_wkt(wkt)

    # 3) DataArray with WKT in attrs (less common)
    if isinstance(obj, xr.DataArray):
        wkt = obj.attrs.get("spatial_ref") or obj.attrs.get("crs_wkt")
        if wkt:
            return CRS.from_wkt(wkt)

    raise ValueError("No CRS found on object and no usable WKT in attributes.")


def _to_crs_obj(crs_like: Any) -> CRS:
    """Accept str | rasterio.CRS and return rasterio.CRS."""
    if isinstance(crs_like, CRS):
        return crs_like
    return CRS.from_string(str(crs_like))


def _resolve_dynamic_da(
    name: str,
    ds_by_store: dict[str, xr.Dataset],
    var_name_map: dict[str, dict[str, Any]],
) -> xr.DataArray:
    """
    Build the requested dynamic DataArray (with optional .sel()) and

    rename to its stats_key for consistent normalization keys.
    """
    meta = var_name_map[name]
    store_key = meta["store"]
    var = meta["var"]
    sel = meta.get("sel") or {}
    stats_key = meta.get("stats_key", name)

    ds = ds_by_store[store_key]
    da = ds[var]
    if sel:
        da = da.sel(**sel)
    # give a stable name used by your stats JSON
    da = da.rename(stats_key)
    return da


def extract_floodid(path: str) -> str:
    """Parse the floodID from a path of form .../DFO_<floodID>_<start>_<end>/file.tif"""
    dir_name = os.path.basename(os.path.dirname(path))
    parts = dir_name.split("_")
    return parts[1] if len(parts) > 1 else dir_name


def compute_modis_bboxes(
    paths: Iterable[str], input_crs: str, bounds: tuple
) -> dict[str, tuple[float, float, float, float]]:
    """Read each MODIS TIFF, transform its bounds to the input CRS, and return a dict

    mapping floodID to (left, bottom, right, top) for CONUS events.
    Accept any partial overlap with the USA CONUS area.
    """
    # CONUS bounding box in WGS84 (lon/lat)
    # conus_wgs84 = (-124.848974, 24.396308, -66.885444, 49.384358)
    ref_bounds = bounds
    target_crs = input_crs
    bboxes = {}
    for path in paths:
        # extract floodID from directory name
        floodid = extract_floodid(path)
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            src_crs = src.crs
        dst_bounds = (left, bottom, right, top)
        if src_crs != target_crs:
            # reproject bounds if needed
            dst_bounds = transform_bounds(src_crs, target_crs, left, bottom, right, top)
        # Accept any partial intersection with CONUS
        if not (
            dst_bounds[2] < ref_bounds[0]
            or dst_bounds[0] > ref_bounds[2]
            or dst_bounds[3] < ref_bounds[1]
            or dst_bounds[1] > ref_bounds[3]
        ):
            bboxes[floodid] = dst_bounds
    return bboxes


def compute_modis_overlap_bboxes(
    paths: Iterable[str],
    input_crs: str | CRS,
    conus_shp_gpkg_path: str | Path,
    conus_layer: str | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """
    For each MODIS GeoTIFF:

      1) Get raster bounds in native CRS,
      2) Reproject bounds to `input_crs`,
      3) Intersect with CONUS polygon from a GeoPackage (also in `input_crs`),
      4) If intersection exists, store the intersection bbox.

    Returns
    -------
      { flood_id : (minx, miny, maxx, maxy) } in `input_crs`.
    """
    target_crs_rio = _to_rio_crs(input_crs)
    target_crs_pj = _to_pyproj_crs(target_crs_rio)

    conus_geom = _load_conus_geom_gpkg(conus_shp_gpkg_path, target_crs=target_crs_rio, layer=conus_layer)
    out: dict[str, tuple[float, float, float, float]] = {}

    for path in paths:
        floodid = extract_floodid(path)
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            src_crs_rio: CRS = src.crs

        raster_geom = box(left, bottom, right, top)

        if src_crs_rio is not None and src_crs_rio != target_crs_rio:
            src_crs_pj = _to_pyproj_crs(src_crs_rio)
            transformer = Transformer.from_crs(src_crs_pj, target_crs_pj, always_xy=True)
            raster_geom = shp_transform(transformer.transform, raster_geom)

        overlap = raster_geom.intersection(conus_geom)
        if overlap.is_empty:
            continue

        minx, miny, maxx, maxy = overlap.bounds
        out[floodid] = (float(minx), float(miny), float(maxx), float(maxy))

        ## to plot the bounds:
        # ax = plot_overlap_debug(raster_geom, conus_geom, overlap)
        # plt.show()
    return out


def _load_conus_geom_gpkg(
    gpkg_path: str | Path,
    target_crs: str | CRS,
    layer: str | None = None,
):
    """
    Read a CONUS polygon/multipolygon from a GeoPackage, merge all parts,

    and reproject to target_crs. If layer is None, picks the first
    polygon/multipolygon layer.
    """
    gpkg_path = str(gpkg_path)
    target_crs_pj = _to_pyproj_crs(target_crs)

    # Pick layer automatically if not provided
    if layer is None:
        layers = fiona.listlayers(gpkg_path)
        chosen = None
        for lyr in layers:
            with fiona.open(gpkg_path, layer=lyr) as src:
                gtype = (src.schema or {}).get("geometry", "").upper()
                if ("POLYGON" in gtype) or ("MULTIPOLYGON" in gtype):
                    chosen = lyr
                    break
        if chosen is None:
            raise ValueError(f"No POLYGON/MULTIPOLYGON layer found in {gpkg_path}. Layers: {layers}")
        layer = chosen

    with fiona.open(gpkg_path, layer=layer) as src:
        src_crs_wkt = src.crs_wkt
        if src_crs_wkt:
            src_crs_pj = PJCRS.from_wkt(src_crs_wkt)
        else:
            src_crs_pj = PJCRS.from_user_input(src.crs) if src.crs else PJCRS.from_epsg(4326)

        geoms = [shape(feat["geometry"]) for feat in src if feat.get("geometry") is not None]
        if not geoms:
            raise ValueError(f"No geometries found in {gpkg_path}:{layer}")
        geom = unary_union(geoms)

    if src_crs_pj != target_crs_pj:
        transformer = Transformer.from_crs(src_crs_pj, target_crs_pj, always_xy=True)
        geom = shp_transform(transformer.transform, geom)

    return geom


def filter_modis_paths(paths: Iterable[str], bboxes: dict[str, tuple]) -> list[str]:
    """Filter flood events outside CONUS using computed bboxes, log excluded IDs, and return sorted list of valid paths."""
    keep = []
    for path in paths:
        floodid = extract_floodid(path)
        if floodid in bboxes:
            keep.append(path)
    if not keep:
        raise RuntimeError("No MODIS flood events remain within CONUS bounds.")
    return sorted(keep)


def _to_rio_crs(crs_like: str | CRS) -> CRS:
    """
    Normalize a CRS input to rasterio.CRS.

    Accepts 'EPSG:XXXX' strings, WKT strings, or already-CRS objects.
    """
    if isinstance(crs_like, CRS):
        return crs_like
    if isinstance(crs_like, str):
        # rasterio will parse EPSG:, PROJ4, or WKT strings
        return CRS.from_string(crs_like)
    raise TypeError(f"Unsupported CRS type: {type(crs_like)}")


def _to_pyproj_crs(crs_like: str | CRS | PJCRS) -> PJCRS:
    """Converts crt to pyproj style crs"""
    if isinstance(crs_like, PJCRS):
        return crs_like
    if isinstance(crs_like, CRS):
        # rasterio CRS -> pyproj CRS
        return PJCRS.from_wkt(crs_like.to_wkt())
    return PJCRS.from_user_input(crs_like)


def create_master_from_bounds(
    crs_like: str | CRS,
    bounds: tuple[float, float, float, float],  # (minx, miny, maxx, maxy)
    master_path: str | Path,
    resolution: float = 250.0,  # interpreted in the CRS units
    nodata: float = np.nan,
    fill_value: float = 0.0,
    dtype: str = "float32",
    compress: str = "lzw",
) -> None:
    """
    Create a blank GeoTIFF covering `bounds` in `crs_like` with given `resolution`.

    Notes
    -----
    - `resolution` is assumed to be in the same units as the CRS.
      If the CRS is geographic (degrees), resolution is in degrees.
    - Bounds must be in the same CRS.
    """
    master_path = Path(master_path)
    crs = _to_rio_crs(crs_like)
    minx, miny, maxx, maxy = map(float, bounds)

    if not (maxx > minx and maxy > miny):
        raise ValueError(f"Invalid bounds: {bounds}")

    if getattr(crs, "is_geographic", False):
        log.warning(
            "Master grid CRS is geographic (degrees). The 'resolution' "
            "will be interpreted in degrees (not meters)."
        )

    # size and transform (top-left origin)
    width = int(np.ceil((maxx - minx) / float(resolution)))
    height = int(np.ceil((maxy - miny) / float(resolution)))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Computed non-positive raster size: {(height, width)} from bounds={bounds} and res={resolution}"
        )

    transform = from_origin(minx, maxy, float(resolution), float(resolution))

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "IF_SAFER",
    }

    master_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(master_path, "w", **profile) as dst:
        dst.write(np.full((height, width), fill_value, dtype=np.dtype(dtype)), 1)

    log.info(f"→ Master grid written to {master_path} (H={height}, W={width}, CRS={crs.to_string()})")


def _ensure_master_grid(cfg: Any, inputs_dict: dict[str, Any]) -> str | None:
    """
    Ensure a master grid exists using inputs_dict['input_crs'] and inputs_dict['union_bounds'].

    Required:
      - cfg.master_grids_dir (path-like)
      - cfg.params.resolution (numeric)
      - inputs_dict['input_crs']  : str | rasterio.CRS
      - inputs_dict['union_bounds']: (minx, miny, maxx, maxy) in input_crs
    """
    # --- validate cfg fields ---
    try:
        grids_dir = Path(cfg.master_grids_dir)
    except AttributeError as e:
        log.warning(f"Missing cfg.master_grids_dir: {e}")
        return None

    try:
        resolution = float(cfg.params.resolution)
    except (AttributeError, TypeError, ValueError) as e:
        log.warning(f"Invalid cfg.params.resolution: {e}")
        return None

    # --- build target path ---
    grid_path = grids_dir / f"master_{int(resolution)}m.tif"
    try:
        grid_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        log.warning(f"Cannot create directory for master grid ({grid_path.parent}): {e}")
        return None

    if grid_path.exists():
        return str(grid_path)

    # --- pull CRS + bounds from inputs_dict ---
    try:
        ref_crs_like = inputs_dict["input_crs"]
        ref_bounds = inputs_dict["union_bounds"]
    except KeyError as e:
        log.warning(f"inputs_dict missing required key: {e}")
        return None

    # normalize CRS
    try:
        ref_crs = _to_rio_crs(ref_crs_like)
    except (TypeError, CRSError, ValueError) as e:
        log.warning(f"Invalid input CRS in inputs_dict['input_crs']: {e}")
        return None

    # validate bounds
    if (not isinstance(ref_bounds, tuple | list)) or len(ref_bounds) != 4:
        log.warning(f"Invalid bounds (expect 4-tuple): {ref_bounds!r}")
        return None
    try:
        minx, miny, maxx, maxy = map(float, ref_bounds)
    except (TypeError, ValueError) as e:
        log.warning(f"Bounds must be numeric: {ref_bounds!r} ({e})")
        return None

    # --- create the grid ---
    try:
        create_master_from_bounds(
            crs_like=ref_crs,
            bounds=(minx, miny, maxx, maxy),
            master_path=str(grid_path),
            resolution=resolution,
            nodata=np.nan,  # embedded nodata value
            fill_value=255.0,  # actual pixel fill value
            dtype="float32",
            compress="lzw",
        )
        return str(grid_path)
    except (ValueError, TypeError, CRSError, OSError, PermissionError) as e:
        log.warning(f"Could not ensure master grid (skipping --grid): {e}")
        return None


# -------------------------------------------------------------------------
# (Optional) Backward-compatible shim if any legacy code still passes a DA:
# -------------------------------------------------------------------------
def create_master_from_da(
    da,
    master_path: str | Path,
    resolution: float = 250.0,
    nodata: float = np.nan,
    dtype: str = "float32",
    compress: str = "lzw",
) -> None:
    """
    Compatibility wrapper: derive (crs, bounds) from a DataArray and call create_master_from_bounds().

    Requires 'x' and 'y' coordinates on the DataArray.
    """
    # --- validate spatial coordinates ---
    if "x" not in da.coords or "y" not in da.coords:
        raise ValueError("DataArray must have 'x' and 'y' coordinates to derive bounds.")

    # bounds from coordinates
    try:
        minx = float(da.x.min())
        maxx = float(da.x.max())
        miny = float(da.y.min())
        maxy = float(da.y.max())
    except (TypeError, ValueError) as e:
        raise ValueError(f"Could not compute bounds from 'x'/'y' coordinates: {e}") from e

    # --- robust CRS inference with specific exceptions ---
    crs = None

    # 1) primary: your infer_crs helper
    try:
        crs = infer_crs(da)  # should return a rasterio.CRS
    except (MissingCRS, CRSError, ValueError, AttributeError, KeyError):
        crs = None

    # 2) fallback: rioxarray accessor property (should not raise, but guard anyway)
    if crs is None:
        try:
            crs = getattr(da.rio, "crs", None)
        except AttributeError:
            crs = None

    # 3) fallback: WKT-like attrs commonly seen on EO rasters
    if crs is None:
        wkt = da.attrs.get("spatial_ref") or da.attrs.get("esri_pe_string")
        if wkt:
            try:
                crs = CRS.from_wkt(wkt)
            except (CRSError, ValueError, TypeError):
                crs = None

    if crs is None:
        log.warning("No CRS found on DataArray; defaulting to EPSG:4326")
        crs = CRS.from_epsg(4326)

    create_master_from_bounds(
        crs_like=crs,
        bounds=(minx, miny, maxx, maxy),
        master_path=master_path,
        resolution=resolution,
        nodata=nodata,
        dtype=dtype,
        compress=compress,
    )


def _iter_polys(geom):
    """Yield polygons from Polygon/MultiPolygon; ignore empties."""
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
    else:
        # If it's something else (e.g., GeometryCollection), try to pull polygons
        try:
            for g in geom.geoms:
                if isinstance(g, Polygon | MultiPolygon) and not g.is_empty:
                    yield from _iter_polys(g)
        except AttributeError:
            return


def _plot_geom(ax, geom, facecolor=None, edgecolor="k", alpha=0.3, linewidth=1.0, zorder=1, label=None):
    """Plot Polygon/MultiPolygon (with holes) on an axis."""
    for poly in _iter_polys(geom):
        # exterior
        x, y = poly.exterior.xy
        ax.fill(
            x,
            y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
            label=label,
        )
        label = None  # only label first piece
        # holes (interiors)
        for ring in poly.interiors:
            xi, yi = ring.xy
            ax.fill(xi, yi, facecolor="white", edgecolor="none", alpha=1.0, zorder=zorder + 0.1)


def plot_overlap_debug(raster_geom, conus_geom, overlap_geom, ax=None):
    """
    Visualize raster footprint vs CONUS polygon and their intersection.

    IMPORTANT: All geometries must already be in the SAME CRS.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Draw order: CONUS (base), raster (outline), overlap (highlight)
    _plot_geom(
        ax,
        conus_geom,
        facecolor="#CCCCCC",
        edgecolor="#666666",
        alpha=0.4,
        linewidth=0.8,
        zorder=1,
        label="CONUS",
    )
    _plot_geom(
        ax,
        raster_geom,
        facecolor="none",
        edgecolor="#1f77b4",
        alpha=1.0,
        linewidth=1.5,
        zorder=2,
        label="Raster footprint",
    )

    if overlap_geom is not None and not overlap_geom.is_empty:
        _plot_geom(
            ax,
            overlap_geom,
            facecolor="#ff7f0e",
            edgecolor="#d95f02",
            alpha=0.5,
            linewidth=1.0,
            zorder=3,
            label="Overlap",
        )

    # Set bounds with a small margin
    geoms_to_bound = [g for g in [conus_geom, raster_geom, overlap_geom] if g is not None and not g.is_empty]
    if geoms_to_bound:
        union = unary_union(geoms_to_bound)
        minx, miny, maxx, maxy = union.bounds
        dx = (maxx - minx) * 0.05 or 1.0
        dy = (maxy - miny) * 0.05 or 1.0
        ax.set_xlim(minx - dx, maxx + dx)
        ax.set_ylim(miny - dy, maxy + dy)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Build a clean legend (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles, strict=False))
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), loc="best", frameon=True)

    ax.set_title("Raster footprint vs CONUS and Overlap")
    return ax


# ax = plot_overlap_debug(raster_geom, conus_geom, overlap)
# plt.show()
