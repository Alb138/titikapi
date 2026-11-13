import numpy as np
import pytest

from titikapi.data.land import _dem_tile_name, block_to_cells, category_fractions


def test_block_to_cells_groups_pixels_by_coarse_cell():
    block = np.arange(16).reshape(4, 4)
    cells = block_to_cells(block, 2)
    assert cells.shape == (2, 2, 4)
    assert sorted(cells[0, 0]) == [0, 1, 4, 5]
    assert sorted(cells[1, 1]) == [10, 11, 14, 15]


def test_category_fractions():
    # 2x2 coarse cells of 2x2 pixels; categories 0..2
    block = np.array(
        [
            [0, 1, 2, 2],
            [1, 1, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 2],
        ],
        dtype=np.uint8,
    )
    fr = category_fractions(block, 2, 3)
    assert fr.shape == (3, 2, 2)
    np.testing.assert_allclose(fr.sum(0), 1.0)
    assert fr[1, 0, 0] == pytest.approx(0.75)
    assert fr[2, 0, 1] == pytest.approx(0.75)
    assert fr[0, 1, 0] == pytest.approx(1.0)
    assert fr[1, 1, 1] == pytest.approx(0.25)


def test_dem_tile_names():
    assert _dem_tile_name(-1, 101) == "Copernicus_DSM_COG_30_S01_00_E101_00_DEM"
    assert _dem_tile_name(0, 95) == "Copernicus_DSM_COG_30_N00_00_E095_00_DEM"


def test_oni_is_lagged_to_what_was_published():
    import pandas as pd

    from titikapi.data.enso import oni_known_at, parse_oni

    text = (
        " SEAS  YR   TOTAL   ANOM\n"
        "  JJA 2019  27.5   0.10\n"
        "  JAS 2019  27.4   0.20\n"
        "  ASO 2019  27.3   0.30\n"
        "  NDJ 2019  27.0  -0.40\n"
    )
    oni = parse_oni(text)
    assert oni[pd.Timestamp("2019-09-01")] == 0.20  # JAS ends in September
    assert oni[pd.Timestamp("2020-01-01")] == -0.40  # NDJ 2019 ends in January 2020
    # on 15 Oct 2019 the newest published season is JAS (ended September), not ASO
    out = oni_known_at(pd.DatetimeIndex(["2019-10-15", "2019-11-01", "2019-12-01"]), oni)
    assert out[0] == 0.20 and out[1] == 0.30
    assert pd.isna(out[2])  # SON not in this toy file
