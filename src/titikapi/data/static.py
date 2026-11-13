"""Static (time-invariant) layers: province raster and the land mask of the study region."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from rasterio.features import rasterize

from titikapi.config import PROVINCES, RAW_DIR
from titikapi.grid import Grid

# geoBoundaries gbOpen IDN ADM1 (OSM-derived, ODbL), pinned to a release commit for reproducibility.
PROVINCES_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM1/"
    "geoBoundaries-IDN-ADM1_simplified.geojson"
)
PROVINCES_PATH = RAW_DIR / "boundaries" / "idn_adm1.geojson"


def download_provinces(dest: Path = PROVINCES_PATH) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(PROVINCES_URL, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def load_provinces() -> gpd.GeoDataFrame:
    """The study-region provinces, with `province_id` = 1..15 in config.PROVINCES order."""
    gdf = gpd.read_file(download_provinces())
    missing = set(PROVINCES) - set(gdf["shapeName"])
    if missing:
        raise ValueError(f"provinces not found in boundary file: {sorted(missing)}")
    gdf = gdf[gdf["shapeName"].isin(PROVINCES)].copy()
    gdf["province_id"] = gdf["shapeName"].map({n: i + 1 for i, n in enumerate(PROVINCES)})
    return gdf.sort_values("province_id").reset_index(drop=True)


def province_raster(grid: Grid, provinces: gpd.GeoDataFrame | None = None) -> np.ndarray:
    """uint8 raster of province ids (0 = outside the study region).

    Pass 1 marks every cell a province touches, so coastal cells with some land are kept.
    Pass 2 then assigns cells by their centre point, so border cells go to the province that
    contains the centre.
    """
    gdf = load_provinces() if provinces is None else provinces
    shapes = list(zip(gdf.geometry, gdf["province_id"], strict=True))
    kwargs = dict(out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8")
    touched = rasterize(shapes, all_touched=True, **kwargs)
    centred = rasterize(shapes, all_touched=False, **kwargs)
    return np.where(centred > 0, centred, touched).astype(np.uint8)
