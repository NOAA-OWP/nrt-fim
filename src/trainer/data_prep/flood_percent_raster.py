import os
import subprocess
from pathlib import Path

import click
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio import shutil as rio_shutil
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds, bounds as window_bounds

rasterio.env.logging.getLogger("rasterio._env").setLevel("ERROR")


def calculate_flood_percentage(raster: xr.DataArray, target_resolution: int | float = 250) -> xr.DataArray:
    """Converts a flood extent raster to a raster where each pixel represents % of pixels flooded when the dataset is resampled.

    Args:
        raster (xr.DataArray): Raster of flood extent (binary)
        target_resolution (int | float, optional): Output raster cell resolution to resample to. Defaults to 250.

    Returns
    -------
        xr.DataArray: Resampled raster representing percent flooded
    """
    # Get the current resolution and dimensions
    current_res_x, current_res_y = raster.rio.resolution()
    current_width = raster.rio.width
    current_height = raster.rio.height

    # Calculate the scale factor
    scale_x = abs(target_resolution / current_res_x)
    scale_y = abs(target_resolution / current_res_y)

    # Calculate new dimensions
    new_width = int(current_width / scale_x)
    new_height = int(current_height / scale_y)

    # Create an empty array for our percentages
    percentages = np.zeros((raster.rio.count, new_height, new_width))

    # Get the data as a numpy array
    data = raster.values

    # For each band
    for b in range(1):   #(raster.rio.count):    only for band 0
        # Get the band data
        band_data = data[b]

        # Identify pixels with value 1 (flooded or permanent water)
        binary_mask = (band_data == 1).astype(np.float32)

        # Calculate percentage for each block
        for i in range(new_height):
            for j in range(new_width):
                # Calculate corresponding indices in the original raster
                start_y = int(i * scale_y)
                end_y = int(min((i + 1) * scale_y, current_height))
                start_x = int(j * scale_x)
                end_x = int(min((j + 1) * scale_x, current_width))

                # Extract the block from the original data
                block = binary_mask[start_y:end_y, start_x:end_x]

                # Calculate total number of valid pixels in the block
                total_pixels = block.size

                # Skip if no valid pixels (avoid division by zero)
                if total_pixels == 0:
                    percentages[b, i, j] = np.nan
                    continue

                # Calculate percentage of flooded pixels
                flood_count = np.sum(block)
                percentages[b, i, j] = (flood_count / total_pixels) * 100

    # Create a new raster with the percentages
    try:
        percentage_raster = raster.rio.reproject(
            raster.rio.crs, shape=(new_height, new_width), resampling=Resampling.nearest
        )
    except ZeroDivisionError as e:
        raise ValueError(
            "There is a problem with the target resolution. Are you using a geographic CRS with resolutions < 1 degree? Try a different resolution."
        ) from e

    # Replace the values with our percentages
    percentage_raster.values = percentages

    return percentage_raster


@click.command(
    name="Generate Flood Percent",
    help="A tool to convert binary flood extent rasters to percent of cell flooded at new resolution.",
)
@click.argument(
    "input_path",
    nargs=1,
    type=click.Path(exists=True),
    required=True,
)
@click.argument(
    "output_path",
    nargs=1,
    type=click.Path(),
    required=True,
)
@click.option("--grid", "-g", type=click.Path(exists=True), help="Grid to align raster to. CRS will be used.")
@click.option(
    "--crs",
    "-c",
    type=int,
    help="CRS to reproject to in EPSG number. If grid is given, grid CRS takes priority",
)
@click.option("--resolution", "-r", type=float, help="Pixel resolution in units of output CRS")
@click.option("--overwrite", "-o", is_flag=True, help="Flag to overwrite output file", default=False)
def generate_flood_percent(
    input_path: str,
    output_path: str,
    grid: str = None,
    crs: int = None,
    resolution: int | float = None,
    overwrite: bool = False,
):
    """A script to convert a binary flood extent raster to raster with percent flooded per pixel, generally at a new resolution. When using `--grid` argument, the raster will be re-aligned to a new grid.

    Args:
        input_path (str): Absolute path to input file
        output_path (str): Absolute path to output file. Temporary files will be written to this directory and deleted at end of script.
        grid (str, optional): Absolute path to grid to align output to. Defaults to None.
        crs (int, optional): EPSG projection written as int value (e.g. 4326). If none specified and grid, grid CRS will be used.
                If none specified and no grid, input CRS will be used. Defaults to None.
        resolution (int, optional): Pixel resolution in units of output CRS. If none specified and grid, grid resolution will be used.
                If none specified and no grid, input x-dimension resolution will be used. Defaults to None.
        overwrite (bool, optional): Flag to overwrite existing file output. Defaults to False.
    """
    # save temporary paths
    dir = Path(output_path).parent
    temp_ras = dir / "temp_raster.tif"
    temp_ras_aligned = dir / "temp_raster_aligned.tif"

    ras = rioxarray.open_rasterio(input_path)
    # # ## some MODIS file don't have crs. so I just reproject them to epsg:4326 (like other MODIS crs)
    # ras = ras.rio.reproject("EPSG:4326")
    # input_file_name = "_" + str(Path(input_path).name)
    # input_path_4326 = dir / input_file_name
    # ras.rio.to_raster(input_path_4326)

    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(
            "Overwrite flag not set and output file path exists. "
            "Please use --overwrite flag or remove output file."
        )

    # re-project if CRS given and no grid defined
    if crs and not grid:
        ras = ras.rio.reproject(crs)
        print("Reprojected raster to new CRS")

    # get the grid information
    if grid:
        with rasterio.open(grid) as src:
            grid_width, grid_height = src.width, src.height
            grid_transform = src.transform
            grid_crs = src.crs

        # reproject to grid CRS
        ras = ras.rio.reproject(grid_crs)
        ras.rio.write_crs(grid_crs, inplace=True)
        print("Reprojected raster to grid CRS")

    # set target resolution to input else use the raster's x-res
    # if the raster had non-square cells, the output will be square
    target_resolution = resolution if resolution else ras.rio.resolution()[0]

    flood_percent = calculate_flood_percentage(ras, target_resolution=target_resolution)
    flood_percent.rio.to_raster(temp_ras)

    # re-align and clip to grid if specified
    if grid:
        vrt_options = {
            "resampling": Resampling.nearest,  # nearest resampling to preserve % value
            "crs": grid_crs,
            "transform": grid_transform,
            "height": grid_height,
            "width": grid_width,
        }
        # use warped VRT to resample and transform raster to grid
        # this will be the full extent of grid
        with rasterio.open(temp_ras) as src:
            with WarpedVRT(src, dtype="float32", **vrt_options) as vrt:
                rio_shutil.copy(vrt, temp_ras_aligned, driver="GTiff", tiled="YES", compress="LZW")

        # clip flood back to original extent keeping new grid alignment
        with rasterio.open(input_path) as src_in, rasterio.open(temp_ras_aligned) as src_aln:
            # reproject original MODIS bounds to the aligned/grid CRS
            L, B, R, T = transform_bounds(src_in.crs, src_aln.crs, *src_in.bounds, densify_pts=21)

            # snap those bounds to the aligned pixel grid
            win = from_bounds(L, B, R, T, transform=src_aln.transform)
            Ls, Bs, Rs, Ts = window_bounds(win, transform=src_aln.transform)

        cmd = [
            "rio", "clip",
            str(temp_ras_aligned),
            str(output_path),
            "--bounds", f"{Ls} {Bs} {Rs} {Ts}",  # ONE string
            # "--overwrite",                      # add if you want
        ]
        # cmd = f"rio clip {temp_ras_aligned} {output_path} --like {input_path}".split()
        cmd += ["--overwrite"] if overwrite else []

        subprocess.call(cmd)
        print("Clipped raster to final extent")
    else:
        flood_percent.rio.to_raster(output_path)

    for f in [temp_ras, temp_ras_aligned]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed temporary {f}")


if __name__ == "__main__":
    generate_flood_percent()

