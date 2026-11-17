"""Compact U-Net mapping daily input maps to 7 fire-probability maps (lead days 1-7)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def weather_sampling_grid(
    weather_lat: np.ndarray, weather_lon: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> torch.Tensor:
    """(1, ny, nx, 2) grid for F.grid_sample: where each 0.1 deg cell centre sits in the
    0.25 deg weather array, in grid_sample's normalised [-1, 1] coordinates (align_corners=True:
    -1 and 1 are the centres of the first and last weather points)."""

    def norm(v, axis):
        return 2 * (v - axis[0]) / (axis[-1] - axis[0]) - 1

    gy, gx = np.meshgrid(norm(lats, weather_lat), norm(lons, weather_lon), indexing="ij")
    return torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32))


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )


class FireUNet(nn.Module):
    """Inputs: grid maps (B, C, ny, nx) and native-resolution weather (B, Cw, h, w).

    Weather is bilinearly resampled onto the grid with a fixed sampling grid, concatenated with
    the grid maps, and passed through a `depth`-level U-Net. Output: logits (B, 7, ny, nx).
    """

    def __init__(
        self,
        grid_channels: int,
        weather_channels: int,
        sampling_grid: torch.Tensor,
        out_channels: int = 7,
        base: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("sampling_grid", sampling_grid, persistent=False)
        self.depth = depth
        widths = [base * 2**i for i in range(depth)]
        self.down = nn.ModuleList()
        cin = grid_channels + weather_channels
        for w in widths:
            self.down.append(ConvBlock(cin, w))
            cin = w
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2)
        self.drop = nn.Dropout2d(dropout)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.dec.append(ConvBlock(2 * w, w))
            cin = w
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(
        self,
        grid: torch.Tensor,
        weather: torch.Tensor,
        crop: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """`crop` = (row0, col0, height, width): train on a window of the region (augmentation)."""
        b = grid.shape[0]
        sg = self.sampling_grid.expand(b, -1, -1, -1).to(weather.dtype)
        wx = F.grid_sample(weather, sg, mode="bilinear", align_corners=True)
        x = torch.cat([grid, wx], dim=1)
        if crop is not None:
            r0, c0, ch, cw = crop
            x = x[..., r0 : r0 + ch, c0 : c0 + cw]
        h, w = x.shape[-2:]

        mult = 2**self.depth  # pad so every pooling level divides evenly
        ph, pw = (-h) % mult, (-w) % mult
        x = F.pad(x, (0, pw, 0, ph))
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.drop(self.bottleneck(x))
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)[..., :h, :w]
