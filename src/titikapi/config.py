"""Project-wide constants: region, grid resolution, date splits, paths."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CUBE_DIR = DATA_DIR / "cube"


@dataclass(frozen=True)
class Region:
    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    res: float  # degrees


REGION = Region(
    "sumatra_kalimantan", lon_min=95.0, lon_max=119.5, lat_min=-6.5, lat_max=6.5, res=0.1
)

# geoBoundaries ADM1 names (English) of the 15 provinces on Sumatra, Kalimantan and nearby islands.
PROVINCES = (
    "Aceh",
    "North Sumatra",
    "West Sumatra",
    "Riau",
    "Riau Islands",
    "Jambi",
    "Bengkulu",
    "South Sumatra",
    "Bangka-Belitung Islands",
    "Lampung",
    "West Kalimantan",
    "Central Kalimantan",
    "South Kalimantan",
    "East Kalimantan",
    "North Kalimantan",
)

# VIIRS S-NPP science-quality data starts 2012-01-20.
FIRST_DAY = dt.date(2012, 1, 20)
SPLITS = {
    "train": (FIRST_DAY, dt.date(2020, 12, 31)),
    "val": (dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
    "test": (dt.date(2023, 1, 1), dt.date(2026, 12, 31)),
}

LEAD_DAYS = 7  # forecast horizon (days ahead)
FIRE_HISTORY_DAYS = 14  # past days of fire activity fed to the model
