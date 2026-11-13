"""Train the U-Net, calibrate it on validation days, and produce forecasts for evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from titikapi.config import LEAD_DAYS, ROOT
from titikapi.grid import Grid
from titikapi.models.dataset import ARRAYS_DIR, Arrays, FireDataset
from titikapi.models.metrics import average_precision
from titikapi.models.unet import FireUNet, weather_sampling_grid

MODELS_DIR = ROOT / "models"


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    base: int = 32
    depth: int = 4
    dropout: float = 0.1
    pos_weight: float = 8.0  # fire cells are ~1-3% of land cells
    patience: int = 6  # epochs without validation improvement before stopping
    workers: int = 2
    amp: bool = True
    seed: int = 0


def build_model(a: Arrays, cfg: TrainConfig) -> FireUNet:
    grid = Grid()
    sg = weather_sampling_grid(a.weather_lat, a.weather_lon, grid.lats, grid.lons)
    return FireUNet(
        a.n_grid_channels, a.weather.shape[1], sg, LEAD_DAYS, cfg.base, cfg.depth, cfg.dropout
    )


def masked_bce(logits, y, mask, pos_weight: float) -> torch.Tensor:
    """BCE over land cells only; positives up-weighted."""
    w = mask[:, None].expand_as(y).float()
    loss = F.binary_cross_entropy_with_logits(
        logits, y.float(), weight=w, pos_weight=torch.tensor(pos_weight, device=y.device)
    )
    return loss * w.numel() / w.sum().clamp(min=1)


def _loader(path: Path, idx, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        FireDataset(path, idx),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.workers,
        persistent_workers=cfg.workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_logits(model, path: Path, idx: np.ndarray, cfg: TrainConfig, device) -> np.ndarray:
    """(D, 7, N) float16 logits for land cells, in `idx` order."""
    model.eval()
    mask = torch.from_numpy(np.load(path / "mask.npy")).to(device)
    out = []
    for x, w, _, _, _ in _loader(path, idx, cfg, shuffle=False):
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True), w.to(device, non_blocking=True))
        out.append(logits.float()[:, :, mask].half().cpu().numpy())
    return np.concatenate(out)


def _val_ap(logits: np.ndarray, a: Arrays, idx: np.ndarray, leads=(1, LEAD_DAYS)) -> dict:
    return {f"ap_lead{d}": average_precision(a.fire_on[idx + d], logits[:, d - 1]) for d in leads}


def train(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig,
    name: str = "unet",
    path: Path = ARRAYS_DIR,
) -> Path:
    """Train with early stopping on mean validation PR-AUC (leads 1 and 7). Returns the
    checkpoint path of the best epoch."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = Arrays.load(path)
    model = build_model(a, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = _loader(path, train_idx, cfg, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(loader), pct_start=0.1
    )
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    log = []
    best, best_epoch = -np.inf, -1
    print(f"{name}: {n_params / 1e6:.2f}M parameters, {len(train_idx)} train days, {device}")
    for epoch in range(cfg.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, w, y, _, _ in loader:
            x, w, y = (v.to(device, non_blocking=True) for v in (x, w, y))
            land = x[:, -1] > 0.5  # last grid channel is the land mask
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                logits = model(x, w)
            loss = masked_bce(logits.float(), y, land, cfg.pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(x)
            n += len(x)

        val_logits = predict_logits(model, path, val_idx, cfg, device)
        scores = _val_ap(val_logits.astype(np.float32), a, val_idx)
        score = float(np.mean(list(scores.values())))
        row = {"epoch": epoch + 1, "train_loss": total / n, **scores, "secs": time.time() - t0}
        log.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
        elif epoch - best_epoch >= cfg.patience:
            print(f"  early stop: no improvement for {cfg.patience} epochs")
            break
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    return out_dir / "best.pt"


class PlattCalibrator:
    """Per-lead logistic recalibration p = sigmoid(a * logit + b), fitted on validation days.

    Training with up-weighted positives inflates probabilities; this maps them back to observed
    frequencies without changing the ranking (so PR-AUC and top-k coverage are unaffected).
    """

    def fit(self, logits: np.ndarray, y: np.ndarray, max_rows: int = 2_000_000, seed: int = 0):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(seed)
        self.params = []
        for d in range(logits.shape[1]):
            z = logits[:, d].astype(np.float32).ravel()
            t = y[:, d].ravel()
            take = rng.choice(len(z), size=min(max_rows, len(z)), replace=False)
            lr = LogisticRegression(C=1e6).fit(z[take, None], t[take])
            self.params.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
        return self

    def transform(self, logits: np.ndarray, lead: int) -> np.ndarray:
        a, b = self.params[lead - 1]
        return (1 / (1 + np.exp(-(a * logits.astype(np.float32) + b)))).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.params))

    @classmethod
    def load(cls, path: Path) -> PlattCalibrator:
        c = cls()
        c.params = [tuple(p) for p in json.loads(path.read_text())]
        return c
