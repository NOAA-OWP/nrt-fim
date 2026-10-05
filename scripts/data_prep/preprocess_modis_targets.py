from __future__ import annotations

import glob
import json
import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import hydra
import rasterio
from omegaconf import DictConfig
from rasterio.crs import CRS

from trainer.data_prep.flood_percent_raster import generate_flood_percent
from trainer.data_prep.read_inputs import read_selected_inputs
from trainer.utils.geo_utils import _ensure_master_grid, compute_modis_overlap_bboxes, filter_modis_paths

log = logging.getLogger(__name__)


def _discover_raw_modis(cfg: DictConfig) -> list[str]:
    """Find raw MODIS TIFFs under DFO_* folders."""
    modis_dir = cfg.data_sources.dfo_modis_dir
    pattern = os.path.join(modis_dir, "DFO_*", "*.tif")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No MODIS TIFF files found at {pattern}")
    return paths


# ───────────────────────────────────────────────────────────────────────────────
# Ensure a CRS on raw GeoTIFFs (assign EPSG:4326 if missing)
# ───────────────────────────────────────────────────────────────────────────────
def _ensure_crs_if_missing(
    input_path: str | Path,
    default_epsg: str = "EPSG:4326",
    in_place: bool = False,
) -> str:
    """
    If the GeoTIFF has no CRS, assign `default_epsg`.

    By default, writes a sibling file with suffix *_crs4326.tif and returns its path.
    Set in_place=True to replace the original (via atomic rename).
    """
    p = Path(input_path)
    with rasterio.open(p) as src:
        if src.crs is not None:
            return str(p)

        profile = src.profile.copy()
        profile.update(crs=CRS.from_string(default_epsg))
        data = src.read()  # read all bands into memory

    if in_place:
        tmp = p.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)
        backup = p.with_suffix(".backup_no_crs.tif")
        p.rename(backup)
        tmp.rename(p)
        log.info(f"[crs ] assigned {default_epsg} in-place to {p.name} (backup saved as {backup.name})")
        return str(p)
    else:
        fixed = p.with_name(p.stem + "_crs4326.tif")
        with rasterio.open(fixed, "w", **profile) as dst:
            dst.write(data)
        log.info(f"[crs ] assigned {default_epsg} to {p.name} → {fixed.name}")
        return str(fixed)


def preprocess_modis_batch(
    cfg: DictConfig,
    raw_paths: Iterable[str],
    grid: str | None = None,
    cleanup_temp: bool = True,  # delete *_crs4326.tif after success
    assign_in_place: bool = False,  # set True to modify raws directly
) -> list[str]:
    """
    For each raw MODIS TIFF, ensure CRS (assign EPSG:4326 if missing), then

    generate a flood-percent raster by invoking the click command.

    Returns the list of produced output paths.
    """
    resolution = str(cfg.params.resolution)
    overwrite_flag = bool(cfg.params.get("overwrite_click", False))
    out_paths: list[str] = []

    for raw_input in raw_paths:
        # 0) Ensure CRS on the raw file
        input_to_use = _ensure_crs_if_missing(raw_input, default_epsg="EPSG:4326", in_place=assign_in_place)
        temp_created = input_to_use != str(raw_input)

        # 1) Compute output path (same layout you had)
        file_name = Path(raw_input).name
        dir0 = Path(raw_input).parents[2]
        folder_name0 = Path(raw_input).parents[0].name
        folder_name1 = Path(raw_input).parents[1].name
        output_path = dir0.joinpath(folder_name1 + "_perc_regrid", folder_name0, file_name)

        if output_path.exists():
            log.info(f"[skip] exists: {output_path}")
            out_paths.append(str(output_path))
            # optional cleanup of temp CRS file
            if temp_created and cleanup_temp:
                Path(input_to_use).unlink(missing_ok=True)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 2) Build CLI args and run
        args = [input_to_use, str(output_path), "--resolution", resolution]
        if overwrite_flag:
            args.append("--overwrite")
        if grid:
            args += ["--grid", grid]

        log.info(f"[run ] flood-percent: {input_to_use} -> {output_path}")
        try:
            generate_flood_percent.main(args, standalone_mode=False)
            if output_path.exists():
                out_paths.append(str(output_path))
            else:
                # No overlap → clean empty dir
                out_dir = output_path.parent
                if out_dir.exists() and not any(out_dir.iterdir()):
                    shutil.rmtree(out_dir)
        except SystemExit as e:
            if e.code != 0:
                raise RuntimeError(f"generate_flood_percent failed on {raw_input}") from None
        finally:
            # 3) Cleanup temporary *_crs4326.tif if we created it
            if temp_created and cleanup_temp:
                try:
                    Path(input_to_use).unlink(missing_ok=True)
                except PermissionError as ce:
                    log.warning(f"Could not remove temp file {input_to_use}: {ce}")

        log.info(f"[done] {output_path}")

    return out_paths


@hydra.main(
    version_base="1.3",
    config_path="../../config",
    config_name="training_config",
)
def main(cfg: DictConfig):
    """
    Reading inputs dictionary, compute bounds, crs, master grid, and process satellite images and convert them all to master grid

    :param cfg: configuration file
    :return: None. saves the processed images in a predefined path
    """
    # 1) Discover raw targets
    raw_paths = _discover_raw_modis(cfg)
    log.info(f"Discovered {len(raw_paths)} raw MODIS TIFFs")

    # ─────────────────────────────────────────────────────────────
    # NEW: Filter raws to those within CONUS bounds (precip CRS)
    # inputs_dict has the following keys:
    # 'dyn_ds': dynamic stores that dynamic variables are written from,
    # 'dyn_vars': dynamic inputs variables,
    # 'crs_by_store': crs of each store,
    # 'input_crs': the reference crs that is used for future mappings,
    # 'hourly_time': time in np.datetime64 format for hourly variables,
    # 'three_hourly_time': time in np.datetime64 format for three-hourly variables,
    # 'static_paths': the paths that static inputs will be read from,
    # 'bounds_per_feature': list of bound per feature based on reference coordinate system (input_crs),
    # 'union_bounds': The minimum bound of all bounds pre feature. --> the reference bound
    # ─────────────────────────────────────────────────────────────
    inputs_dict = read_selected_inputs(cfg=cfg, compute_bounds=True)
    ref_crs = inputs_dict["input_crs"]

    # For bbox computation we need a CRS on every file;
    # create temporary CRS-assigned copies where missing.
    mapping_bbox_to_raw: dict[str, str] = {}
    paths_for_bbox: list[str] = []
    temp_bbox_files: list[Path] = []

    for p in raw_paths:
        ensured = _ensure_crs_if_missing(p, default_epsg="EPSG:4326", in_place=False)
        mapping_bbox_to_raw[ensured] = p
        paths_for_bbox.append(ensured)
        if ensured != str(p):
            temp_bbox_files.append(Path(ensured))

    bboxes = compute_modis_overlap_bboxes(
        paths=paths_for_bbox, input_crs=ref_crs, conus_shp_gpkg_path=cfg["data_sources"]["conus_geom"]
    )
    kept_for_bbox = filter_modis_paths(paths_for_bbox, bboxes)
    kept_raw_paths = [mapping_bbox_to_raw[k] for k in kept_for_bbox]

    # Clean up temporary *_crs4326.tif used only for filtering
    for t in temp_bbox_files:
        try:
            t.unlink(missing_ok=True)
        except PermissionError as e:
            log.warning(f"Could not remove temp bbox file {t}: {e}")

    log.info(f"Filtered to {len(kept_raw_paths)} files within CONUS bounds")

    # 2) (Optional) ensure a master grid to pass as --grid
    grid_path = _ensure_master_grid(cfg, inputs_dict=inputs_dict)

    # 3) Preprocess batch (assign EPSG:4326 where missing during processing as well)
    produced = preprocess_modis_batch(cfg, kept_raw_paths, grid=grid_path)
    log.info(f"Produced {len(produced)} flood-percent files")

    # 4) Write manifest for downstream steps
    manifest = Path(cfg["data_sources"]["dfo_modis_dir_preprocessed"]) / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as json_file:
        json.dump(produced, json_file, indent=4)
    print(f"Wrote manifest: {manifest}  ({len(produced)} paths)")


if __name__ == "__main__":
    main()
