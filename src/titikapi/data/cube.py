"""Loading the built cubes into compact in-memory arrays for modelling and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from titikapi.config import CUBE_DIR


@dataclass
class FireMatrix:
    """Daily fire occurrence for the study-region land cells only.

    `on[t, i]` is True when land cell i had at least one detection on day t. Cells are the
    `mask` cells in row-major order, so `grid_values[mask]` maps a full grid to this layout.
    """

    time: pd.DatetimeIndex
    on: np.ndarray  # (T, N) bool
    mask: np.ndarray  # (ny, nx) bool
    province: np.ndarray  # (N,) uint8, 1..15

    @property
    def n_cells(self) -> int:
        return self.on.shape[1]

    def to_grid(self, values: np.ndarray, fill=np.nan) -> np.ndarray:
        """(..., N) cell values -> (..., ny, nx) grid, `fill` outside the region."""
        out = np.full(values.shape[:-1] + self.mask.shape, fill, dtype=np.result_type(values, fill))
        out[..., self.mask] = values
        return out


def load_static(cube_dir: Path = CUBE_DIR) -> xr.Dataset:
    return xr.open_zarr(cube_dir / "static.zarr", consolidated=False).load()


def load_fire_matrix(cube_dir: Path = CUBE_DIR) -> FireMatrix:
    static = load_static(cube_dir)
    mask = static["mask"].values
    fire = xr.open_zarr(cube_dir / "fire.zarr", consolidated=False)
    t = fire.sizes["time"]
    on = np.zeros((t, int(mask.sum())), dtype=bool)
    for i in range(0, t, 366):  # a year at a time keeps peak memory low
        on[i : i + 366] = fire["fire_count"].isel(time=slice(i, i + 366)).values[:, mask] > 0
    return FireMatrix(
        time=pd.DatetimeIndex(fire["time"].values),
        on=on,
        mask=mask,
        province=static["province_id"].values[mask],
    )


def recent_fire(on: np.ndarray, window: int = 14) -> np.ndarray:
    """recent[t, i] = any fire in cell i during days [t - window + 1, t]."""
    if on.shape[0] >= np.iinfo(np.int16).max:
        raise ValueError("series too long for int16 cumulative counts")
    cs = np.cumsum(on, axis=0, dtype=np.int16)
    win = cs.copy()
    win[window:] -= cs[:-window]
    return win > 0
