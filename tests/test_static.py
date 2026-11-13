import pytest

from titikapi.config import PROVINCES
from titikapi.data.static import province_raster
from titikapi.grid import Grid

pytestmark = pytest.mark.network  # downloads the boundary file on first run


@pytest.fixture(scope="module")
def raster():
    grid = Grid()
    return grid, province_raster(grid)


@pytest.mark.parametrize(
    "lon, lat, province",
    [
        (101.45, 0.53, "Riau"),  # Pekanbaru
        (113.92, -2.21, "Central Kalimantan"),  # Palangka Raya
        (95.32, 5.55, "Aceh"),  # Banda Aceh
        (104.76, -2.99, "South Sumatra"),  # Palembang
        (117.15, 3.0, "North Kalimantan"),
    ],
)
def test_known_cities_land_in_their_province(raster, lon, lat, province):
    grid, r = raster
    row, col = grid.point_to_cell([lon], [lat])
    assert PROVINCES[r[row[0], col[0]] - 1] == province


@pytest.mark.parametrize(
    "lon, lat",
    [
        (106.85, -6.2),  # Jakarta (Java, not in the study region)
        (110.35, 1.55),  # Kuching, Malaysia
        (110.0, -5.0),  # Java Sea
        (101.7, 3.15),  # Kuala Lumpur
    ],
)
def test_outside_points_are_masked(raster, lon, lat):
    grid, r = raster
    row, col = grid.point_to_cell([lon], [lat])
    assert r[row[0], col[0]] == 0


def test_all_provinces_present(raster):
    _, r = raster
    assert set(r[r > 0].ravel()) == set(range(1, len(PROVINCES) + 1))
