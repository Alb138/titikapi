import numpy as np
import pandas as pd
import pytest

from titikapi.config import REGION
from titikapi.grid import Grid


@pytest.fixture
def grid() -> Grid:
    return Grid(REGION)


def test_shape_and_centers(grid: Grid):
    assert grid.shape == (130, 245)
    # north-up raster: first row is the northernmost
    assert grid.lats[0] == pytest.approx(6.45)
    assert grid.lats[-1] == pytest.approx(-6.45)
    assert grid.lons[0] == pytest.approx(95.05)
    assert grid.lons[-1] == pytest.approx(119.45)


@pytest.mark.parametrize(
    "lon, lat, expected",
    [
        (95.0, 6.5, (0, 0)),  # top-left corner is inside (half-open interval)
        (95.05, 6.45, (0, 0)),
        (100.0, 0.0, (65, 50)),  # exact interior boundaries land in the lower/right cell
        (101.4499, 0.5501, (59, 64)),
        (119.49, -6.49, (129, 244)),
    ],
)
def test_point_to_cell_inside(grid: Grid, lon, lat, expected):
    rows, cols = grid.point_to_cell(np.array([lon]), np.array([lat]))
    assert (int(rows[0]), int(cols[0])) == expected


@pytest.mark.parametrize(
    "lon, lat",
    [(119.5, 0.0), (94.99, 0.0), (100.0, -6.5), (100.0, 6.51)],
)
def test_point_to_cell_outside(grid: Grid, lon, lat):
    rows, cols = grid.point_to_cell(np.array([lon]), np.array([lat]))
    assert rows[0] == -1 and cols[0] == -1


def test_rasterize_counts_and_sums(grid: Grid):
    lon = np.array([100.01, 100.09, 100.15, 200.0])  # last one is outside the region
    lat = np.array([0.01, 0.05, 0.05, 0.0])
    frp = np.array([1.0, 2.0, 4.0, 100.0])
    counts = grid.rasterize(lon, lat)
    sums = grid.rasterize(lon, lat, values=frp)
    assert counts.shape == grid.shape
    assert counts[64, 50] == 2 and counts[64, 51] == 1
    assert counts.sum() == 3
    assert sums[64, 50] == pytest.approx(3.0) and sums[64, 51] == pytest.approx(4.0)


def test_local_solar_date_moves_night_overpass_to_local_day():
    from titikapi.grid import local_solar_date

    # 18:30 UTC at 105E is 01:30 local solar time on the NEXT calendar day
    utc = pd.to_datetime(["2019-09-14 18:30", "2019-09-15 06:00"])
    lon = np.array([105.0, 105.0])
    out = local_solar_date(utc, lon)
    assert list(out) == [pd.Timestamp("2019-09-15"), pd.Timestamp("2019-09-15")]
