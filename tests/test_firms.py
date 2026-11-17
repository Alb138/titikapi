import io

import numpy as np
import pandas as pd
import pytest

from titikapi.data.firms import clean_detections, daily_fire_grids, read_firms_csv
from titikapi.grid import Grid

CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type
0.55,101.45,330.0,0.4,0.4,2019-09-15,0610,N,VIIRS,n,2,290.0,3.0,D,0
0.56,101.44,340.0,0.4,0.4,2019-09-15,0610,N,VIIRS,h,2,295.0,5.0,D,0
0.55,101.45,320.0,0.4,0.4,2019-09-15,0610,N,VIIRS,l,2,290.0,9.0,D,0
0.55,101.45,320.0,0.4,0.4,2019-09-15,0610,N,VIIRS,n,2,290.0,9.0,D,2
-3.0,138.0,320.0,0.4,0.4,2019-09-15,0410,N,VIIRS,n,2,290.0,9.0,D,0
-2.05,113.05,310.0,0.4,0.4,2019-09-14,1830,N,VIIRS,n,2,290.0,1.5,N,0
"""


@pytest.fixture
def raw() -> pd.DataFrame:
    return read_firms_csv(io.StringIO(CSV))


def test_read_keeps_leading_zero_times(raw):
    assert raw["acq_time"].iloc[0] == "0610"


def test_clean_filters_confidence_type_and_region(raw):
    det = clean_detections(raw, Grid())
    # dropped: low confidence, static source (type 2), outside region (Papua)
    assert len(det) == 3
    assert set(det.columns) >= {"lon", "lat", "frp", "local_date", "row", "col"}


def test_clean_assigns_night_overpass_to_local_day(raw):
    det = clean_detections(raw, Grid())
    night = det[det["lon"] == 113.05].iloc[0]
    assert night["local_date"] == pd.Timestamp("2019-09-15")


def test_daily_grids(raw):
    grid = Grid()
    det = clean_detections(raw, grid)
    ds = daily_fire_grids(det, "2019-09-13", "2019-09-16", grid)
    assert ds["fire_count"].shape == (4, *grid.shape)
    assert ds["fire_count"].dtype == np.int16
    per_day = ds["fire_count"].sum(dim=("lat", "lon")).values
    assert list(per_day) == [0, 0, 3, 0]
    riau = ds.sel(time="2019-09-15", lat=0.55, lon=101.45, method="nearest")
    assert int(riau["fire_count"]) == 2
    assert float(riau["frp_sum"]) == pytest.approx(8.0)


def test_daily_grids_ignores_detections_outside_date_range(raw):
    grid = Grid()
    det = clean_detections(raw, grid)
    ds = daily_fire_grids(det, "2019-09-16", "2019-09-17", grid)
    assert int(ds["fire_count"].sum()) == 0


def test_drop_static_sources_removes_only_flagged_locations():
    import numpy as np
    import pandas as pd

    from titikapi.data.firms import drop_static_sources, static_source_points

    sp = pd.DataFrame(
        {"longitude": [101.0, 102.0, 103.0], "latitude": [1.0, 0.5, -1.0], "type": [2, 0, 1]}
    )
    pts = static_source_points([sp])
    assert np.allclose(sorted(pts[:, 0]), [101.0, 103.0])
    nrt = pd.DataFrame(
        {"longitude": [101.002, 101.02, 102.0, 103.0], "latitude": [1.001, 1.0, 0.5, -1.0]}
    )
    kept = drop_static_sources(nrt, pts)
    assert kept["longitude"].tolist() == [101.02, 102.0]  # 2 km away and a vegetation fire stay
