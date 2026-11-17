import numpy as np
import pandas as pd
import pytest

from titikapi.models.dataset import Arrays, days_since_fire, grid_inputs, quiet_cells, targets

torch = pytest.importorskip("torch")


def _arrays(t=60, ny=6, nx=8, seed=0) -> Arrays:
    rng = np.random.default_rng(seed)
    mask = np.ones((ny, nx), bool)
    mask[0, 0] = False
    n = int(mask.sum())
    on = rng.random((t, n)) < 0.1
    return Arrays(
        time=pd.date_range("2019-06-01", periods=t),
        mask=mask,
        fire_on=on,
        fire_count=on.astype(np.uint8) * 3,
        frp_log=(on * 2.0).astype(np.float16),
        days_since=days_since_fire(on),
        static=rng.standard_normal((4, ny, nx)).astype(np.float32),
        forest_loss=rng.random((25, ny, nx)).astype(np.float32) * 0.01,
        loss_years=np.arange(2001, 2026, dtype=np.int16),
        clim=rng.random((13, n)).astype(np.float32) * 0.05,
        oni=np.linspace(-1, 2, t).astype(np.float32),
        weather=rng.standard_normal((t, 3, 4, 5)).astype(np.float16),
        weather_lat=np.array([1.0, 0.5, 0.0, -0.5], np.float32),
        weather_lon=np.array([100.0, 100.5, 101.0, 101.5, 102.0], np.float32),
    )


def test_days_since_fire():
    on = np.array([0, 1, 0, 0, 1], bool)[:, None]
    assert days_since_fire(on, cap=10)[:, 0].tolist() == [10, 0, 1, 2, 0]


def test_grid_inputs_shape_and_today_first():
    a = _arrays()
    x = grid_inputs(a, 30)
    assert x.shape == (a.n_grid_channels, *a.mask.shape)
    today = np.zeros(a.mask.shape, np.float32)
    today[a.mask] = a.fire_on[30]
    np.testing.assert_array_equal(x[0], today)


def test_grid_inputs_never_look_ahead():
    a = _arrays()
    t = 30
    x = grid_inputs(a, t)
    b = _arrays()
    b.fire_on[t + 1 :] = ~b.fire_on[t + 1 :]
    b.fire_count[t + 1 :] = 200
    b.frp_log[t + 1 :] = 9
    b.days_since[t + 1 :] = 0
    b.oni[t + 1 :] = 5
    np.testing.assert_array_equal(x, grid_inputs(b, t))


def test_forest_loss_uses_only_years_before_issue_year():
    a = _arrays()  # issue year 2019
    x = grid_inputs(a, 30)
    a.forest_loss[a.loss_years >= 2019] = 1.0  # loss in or after the issue year
    np.testing.assert_array_equal(x, grid_inputs(a, 30))


def test_targets_and_quiet_cells():
    a = _arrays()
    y = targets(a, 30)
    assert y.shape == (7, *a.mask.shape)
    assert y[0][a.mask].tolist() == a.fire_on[31].astype(np.uint8).tolist()
    q = quiet_cells(a, 30)
    assert not q[0, 0]  # outside the mask
    assert (q[a.mask] == (a.days_since[30] >= 14)).all()


def test_weather_sampling_grid_aligns_lat_lon():
    from titikapi.models.unet import weather_sampling_grid

    a = _arrays()
    lats = np.array([0.9, 0.25, -0.4])
    lons = np.array([100.1, 101.3, 101.9])
    sg = weather_sampling_grid(a.weather_lat, a.weather_lon, lats, lons)
    field = np.broadcast_to(a.weather_lat[:, None], (4, 5)).astype(np.float32)  # value = latitude
    w = torch.from_numpy(np.ascontiguousarray(field))[None, None]
    out = torch.nn.functional.grid_sample(w, sg, mode="bilinear", align_corners=True)
    np.testing.assert_allclose(out[0, 0, :, 0].numpy(), lats, atol=1e-5)


def test_unet_output_shape_on_odd_grid():
    from titikapi.models.unet import FireUNet, weather_sampling_grid

    a = _arrays(ny=13, nx=21)
    lats = np.linspace(0.9, -0.4, 13)
    lons = np.linspace(100.1, 101.9, 21)
    sg = weather_sampling_grid(a.weather_lat, a.weather_lon, lats, lons)
    model = FireUNet(a.n_grid_channels, 3, sg, base=8, depth=2)
    x = torch.from_numpy(grid_inputs(a, 30))[None]
    w = torch.from_numpy(np.asarray(a.weather[30], np.float32))[None]
    assert model(x, w).shape == (1, 7, 13, 21)


def test_unet_crop_returns_window_logits():
    import torch

    from titikapi.models.unet import FireUNet

    sg = torch.zeros(1, 37, 53, 2)
    model = FireUNet(5, 3, sg, base=8, depth=2)
    out = model(torch.randn(2, 5, 37, 53), torch.randn(2, 3, 9, 11), crop=(3, 7, 20, 30))
    assert out.shape == (2, 7, 20, 30)
