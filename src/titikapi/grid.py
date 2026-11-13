"""The 0.1-degree analysis grid: cell geometry, point-to-cell mapping, rasterization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from affine import Affine

from titikapi.config import REGION, Region

_EPS = 1e-9  # guards against float error when a point sits exactly on a cell edge


class Grid:
    """North-up regular lat/lon grid. Row 0 is the northernmost row, column 0 the westernmost."""

    def __init__(self, region: Region = REGION):
        self.region = region
        self.res = region.res
        self.nx = round((region.lon_max - region.lon_min) / region.res)
        self.ny = round((region.lat_max - region.lat_min) / region.res)
        self.lons = region.lon_min + region.res * (np.arange(self.nx) + 0.5)
        self.lats = region.lat_max - region.res * (np.arange(self.ny) + 0.5)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def transform(self) -> Affine:
        """Affine transform for rasterio (pixel -> lon/lat of the top-left corner)."""
        return Affine(self.res, 0.0, self.region.lon_min, 0.0, -self.res, self.region.lat_max)

    def coords(self) -> dict[str, np.ndarray]:
        return {"lat": self.lats, "lon": self.lons}

    def point_to_cell(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map points to (row, col). Points outside the grid get (-1, -1).

        Cells are half-open: [lon_min, lon_max) x (lat_min, lat_max].
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        col = np.floor((lon - self.region.lon_min) / self.res + _EPS).astype(np.int64)
        row = np.floor((self.region.lat_max - lat) / self.res + _EPS).astype(np.int64)
        inside = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        return np.where(inside, row, -1), np.where(inside, col, -1)

    def rasterize(
        self, lon: np.ndarray, lat: np.ndarray, values: np.ndarray | None = None
    ) -> np.ndarray:
        """Count points per cell, or sum `values` per cell if given."""
        row, col = self.point_to_cell(lon, lat)
        keep = row >= 0
        flat = row[keep] * self.nx + col[keep]
        weights = None if values is None else np.asarray(values, dtype=float)[keep]
        out = np.bincount(flat, weights=weights, minlength=self.ny * self.nx)
        return out.reshape(self.shape)


def local_solar_date(utc: pd.DatetimeIndex | pd.Series, lon: np.ndarray) -> pd.DatetimeIndex:
    """Calendar date in local solar time (UTC + lon/15 h).

    VIIRS overpasses Indonesia around 01:30 and 13:30 local time. Using the UTC date would
    put the 01:30 overpass (≈18:30 UTC) on the previous day, splitting one night across two days.
    """
    utc = pd.DatetimeIndex(utc)
    offset = pd.to_timedelta(np.asarray(lon, dtype=float) / 15.0, unit="h")
    return (utc + offset).normalize()
