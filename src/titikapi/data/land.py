"""Land-surface layers aggregated to the 0.1 deg grid: peat, elevation, forest cover and loss.

- Peat: PEATMAP (Xu et al. 2018, Univ. of Leeds, CC BY 4.0); its Indonesia layer is the
  Ministry of Agriculture peat map (BBSDLP 2011), which predates every training year.
- Elevation: Copernicus DEM GLO-90 (public AWS bucket), read remotely at overview resolution.
- Forest: Hansen et al. Global Forest Change v1.13 (tree cover in 2000, year of loss 2001-2025).
  Loss is stored per year so features can use only years *before* the forecast year.
"""

from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window

from titikapi.config import RAW_DIR
from titikapi.grid import Grid

PEATMAP_ZIP = RAW_DIR / "peatmap" / "Asia.zip"
PEATMAP_LAYER = "Asia/SEA_Peatland.shp"  # Southeast Asia; one multipolygon
HANSEN_DIR = RAW_DIR / "hansen"
HANSEN_VERSION = "GFC-2025-v1.13"
HANSEN_FIRST_LOSS_YEAR = 2001
DEM_URL = "https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"


def block_to_cells(block: np.ndarray, factor: int) -> np.ndarray:
    """(H, W) fine pixels -> (H/factor, W/factor, factor*factor) values of each coarse cell."""
    h, w = block.shape
    out = block.reshape(h // factor, factor, w // factor, factor).swapaxes(1, 2)
    return out.reshape(h // factor, w // factor, factor * factor)


def category_fractions(block: np.ndarray, factor: int, n_categories: int) -> np.ndarray:
    """Fraction of each coarse cell's pixels in each integer category 0..n_categories-1.

    Returns (n_categories, H/factor, W/factor). One bincount pass, so it is fast on 16M pixels.
    """
    h, w = block.shape
    rows, cols = h // factor, w // factor
    cell = (np.arange(h)[:, None] // factor) * cols + (np.arange(w)[None, :] // factor)
    keys = cell.ravel().astype(np.int64) * n_categories + block.ravel()
    counts = np.bincount(keys, minlength=rows * cols * n_categories)
    counts = counts.reshape(rows, cols, n_categories)
    return np.moveaxis(counts / (factor * factor), -1, 0).astype(np.float32)


# --- peat -------------------------------------------------------------------------------------


def peat_fraction(grid: Grid, oversample: int = 20) -> np.ndarray:
    """Fraction of each cell covered by peat, from PEATMAP's Southeast Asia layer."""
    with zipfile.ZipFile(PEATMAP_ZIP) as zf:
        if PEATMAP_LAYER not in zf.namelist():
            raise FileNotFoundError(f"{PEATMAP_LAYER} not in {PEATMAP_ZIP}")
    r = grid.region
    gdf = gpd.read_file(f"zip://{PEATMAP_ZIP.as_posix()}!{PEATMAP_LAYER}")  # ESRI:54034
    gdf = gdf.to_crs(4326).clip((r.lon_min, r.lat_min, r.lon_max, r.lat_max))
    fine = (grid.ny * oversample, grid.nx * oversample)
    transform = from_origin(
        grid.region.lon_min, grid.region.lat_max, grid.res / oversample, grid.res / oversample
    )
    burned = rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty),
        out_shape=fine,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return category_fractions(burned, oversample, 2)[1]


# --- elevation --------------------------------------------------------------------------------


def _dem_tile_name(lat_bottom: int, lon_left: int) -> str:
    ns = "N" if lat_bottom >= 0 else "S"
    ew = "E" if lon_left >= 0 else "W"
    return f"Copernicus_DSM_COG_30_{ns}{abs(lat_bottom):02d}_00_{ew}{abs(lon_left):03d}_00_DEM"


def _read_dem_tile(lat_bottom: int, lon_left: int, sub: int) -> np.ndarray | None:
    """1x1 deg tile averaged to (10*sub, 10*sub) pixels; None for ocean (no tile)."""
    name = _dem_tile_name(lat_bottom, lon_left)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    try:
        with rasterio.Env(**env), rasterio.open(f"/vsicurl/{DEM_URL.format(name=name)}") as src:
            n = 10 * sub
            return src.read(1, out_shape=(n, n), resampling=Resampling.average).astype(np.float32)
    except rasterio.errors.RasterioIOError:
        return None


def elevation_layers(grid: Grid, sub: int = 10, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Mean elevation (m) and roughness (std of `sub` x `sub` sub-cell means) per 0.1 deg cell."""
    r = grid.region
    tiles = [
        (lat, lon)
        for lat in range(int(np.floor(r.lat_min)), int(np.ceil(r.lat_max)))
        for lon in range(int(np.floor(r.lon_min)), int(np.ceil(r.lon_max)))
    ]
    with ThreadPoolExecutor(workers) as pool:
        arrays = list(pool.map(lambda t: _read_dem_tile(*t, sub), tiles))

    ny_fine = (int(np.ceil(r.lat_max)) - int(np.floor(r.lat_min))) * 10 * sub
    nx_fine = (int(np.ceil(r.lon_max)) - int(np.floor(r.lon_min))) * 10 * sub
    mosaic = np.zeros((ny_fine, nx_fine), np.float32)  # ocean = 0 m
    top = int(np.ceil(r.lat_max))
    left = int(np.floor(r.lon_min))
    for (lat, lon), a in zip(tiles, arrays, strict=True):
        if a is not None:
            row = (top - (lat + 1)) * 10 * sub
            col = (lon - left) * 10 * sub
            mosaic[row : row + a.shape[0], col : col + a.shape[1]] = a
    # crop the whole-degree mosaic to the grid box, then aggregate sub x sub -> one cell
    row0 = round((top - r.lat_max) / grid.res) * sub
    col0 = round((r.lon_min - left) / grid.res) * sub
    mosaic = mosaic[row0 : row0 + grid.ny * sub, col0 : col0 + grid.nx * sub]
    cells = block_to_cells(mosaic, sub)
    return cells.mean(-1), cells.std(-1)


# --- forest -----------------------------------------------------------------------------------


def _hansen_path(layer: str, tile: str) -> Path:
    return HANSEN_DIR / f"Hansen_{HANSEN_VERSION}_{layer}_{tile}.tif"


def forest_layers(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Tree cover in 2000 (mean %, 0-100) and loss fraction per year: (n_years, ny, nx).

    Reads each tile in 1x1 deg windows (4000 x 4000 px at 0.00025 deg) to keep memory low.
    """
    r = grid.region
    factor = round(grid.res / 0.00025)  # 400 Hansen pixels per 0.1 deg cell
    n_loss = 26  # lossyear values: 0 = no loss, 1..25 = 2001..2025
    cover = np.zeros(grid.shape, np.float32)
    loss = np.zeros((n_loss - 1, *grid.shape), np.float32)
    for lat_top in (10, 0):
        for lon_left in (90, 100, 110):
            tile = f"{abs(lat_top):02d}{'N' if lat_top >= 0 else 'S'}_{lon_left:03d}E"
            with (
                rasterio.open(_hansen_path("treecover2000", tile)) as tc,
                rasterio.open(_hansen_path("lossyear", tile)) as ly,
            ):
                for lat in np.arange(lat_top, lat_top - 10, -1.0):
                    for lon in np.arange(lon_left, lon_left + 10, 1.0):
                        # the part of this 1-deg block that lies in the grid
                        north, south = min(lat, r.lat_max), max(lat - 1, r.lat_min)
                        west, east = max(lon, r.lon_min), min(lon + 1, r.lon_max)
                        if north <= south or east <= west:
                            continue
                        win = Window(
                            col_off=round((west - lon_left) / 0.00025),
                            row_off=round((lat_top - north) / 0.00025),
                            width=round((east - west) / 0.00025),
                            height=round((north - south) / 0.00025),
                        )
                        rows = slice(
                            round((r.lat_max - north) / grid.res),
                            round((r.lat_max - south) / grid.res),
                        )
                        cols = slice(
                            round((west - r.lon_min) / grid.res),
                            round((east - r.lon_min) / grid.res),
                        )
                        tc_block = tc.read(1, window=win)
                        cover[rows, cols] = block_to_cells(tc_block, factor).mean(-1)
                        fr = category_fractions(ly.read(1, window=win), factor, n_loss)
                        loss[:, rows, cols] = fr[1:]
    return cover, loss
