import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _write_synthetic_arrays(path, t=40):
    from titikapi.grid import Grid
    from titikapi.models.dataset import days_since_fire

    rng = np.random.default_rng(0)
    grid = Grid()
    mask = np.zeros(grid.shape, bool)
    mask[30:90, 40:200] = True
    n = int(mask.sum())
    on = rng.random((t, n)) < 0.03
    arrays = {
        "time": pd.date_range("2019-08-01", periods=t).values.astype("datetime64[D]"),
        "mask": mask,
        "fire_on": on,
        "fire_count": on.astype(np.uint8),
        "frp_log": on.astype(np.float16),
        "days_since": days_since_fire(on),
        "static": rng.standard_normal((4, *grid.shape)).astype(np.float32),
        "forest_loss": np.zeros((25, *grid.shape), np.float32),
        "loss_years": np.arange(2001, 2026, dtype=np.int16),
        "clim": np.full((13, n), 0.02, np.float32),
        "oni": np.zeros(t, np.float32),
        "weather": rng.standard_normal((t, 17, 55, 101)).astype(np.float16),
        "weather_lat": np.linspace(6.75, -6.75, 55).astype(np.float32),
        "weather_lon": np.linspace(94.75, 119.75, 101).astype(np.float32),
    }
    path.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(path / f"{k}.npy", v)


def test_train_one_epoch_and_calibrate(tmp_path, monkeypatch):
    from titikapi.models import train as tr
    from titikapi.models.dataset import Arrays

    _write_synthetic_arrays(tmp_path / "arrays")
    monkeypatch.setattr(tr, "MODELS_DIR", tmp_path / "models")
    cfg = tr.TrainConfig(epochs=1, batch_size=2, base=8, depth=2, workers=0)
    ckpt = tr.train(
        np.arange(14, 24), np.arange(24, 30), cfg, name="smoke", path=tmp_path / "arrays"
    )
    assert ckpt.exists()

    a = Arrays.load(tmp_path / "arrays")
    model = tr.build_model(a, cfg)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    idx = np.arange(24, 30)
    logits = tr.predict_logits(model, tmp_path / "arrays", idx, cfg, torch.device("cpu"))
    assert logits.shape == (6, 7, a.fire_on.shape[1])
    y = np.stack([a.fire_on[idx + d] for d in range(1, 8)], axis=1)
    cal = tr.PlattCalibrator().fit(logits, y)
    p = cal.transform(logits[:, 0], 1)
    assert np.all((p > 0) & (p < 1))
