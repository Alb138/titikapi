# TitikApi: 7-day wildfire forecasts for Sumatra & Kalimantan

## Context

The user wants portfolio projects that help them land a **Data Scientist** role. They are at an advanced level in Python and SQL. They asked for a "cool and interesting deep learning project" about Indonesia's current situation and picked **fire & haze early warning**. This is timely: deforestation rose 66% in 2025, and the 2026 El Niño drought brought fires, haze and water shortages.

The project should show **ML judgment, not just `.fit()`**. That means a simple baseline first, time-based evaluation with no data leakage, error analysis and explanations. It is deployed and scored live on real outcomes, so recruiters see actual forecast performance rather than one number from a fixed test set.

**Outcome:** a public web map showing the daily fire-risk forecast for the next 7 days, a live scorecard of past forecasts against fires that actually happened, a reproducible repo, and a write-up of the results.

Working folder: `C:\Users\Albert Mario Laurent\Projects\Data Science Project` (empty, not yet a git repo).
Machine: RTX 3050 Laptop (4 GB VRAM), uv 0.11, Node 24, git, system Python 3.10. The project uses uv-managed Python 3.12.

## Approach (my recommendation)

Divide the region into a grid and treat each day as a multi-channel "image". A U-Net-style network predicts the probability of fire in every cell for each of the next 1–7 days. Gradient boosting run per cell is the strong baseline that the deep learning model has to beat.

Approaches I ruled out:
- **Gradient boosting only.** It's a fine baseline but not deep learning. It also ignores spatial context such as fire spreading from neighboring cells.
- **Raw satellite imagery fed to a CNN.** That's terabytes of imagery for little gain at a 7-day horizon. The drivers are drought, weather and land use, and those already exist as gridded data.

## Problem definition

- **Grid:** 0.1° (~11 km). The bounding box is lon 95.0–119.5, lat −6.5–6.5, about 245×130 cells, with ocean masked out.
- **Target:** `y[t+d, cell] = 1` if at least one VIIRS active-fire detection (nominal or high confidence) lands in the cell on day t+d, for d = 1..7. The model also predicts "any fire in the next 7 days".
- **Key evaluation slice: *new ignitions*,** meaning cells with no fire in the past 14 days. Simply carrying today's fires forward (persistence) already predicts ongoing fires well. The real skill is predicting new ones.
- **Time splits:** train 2012–2020, validation 2021–2022, test 2023–2026. The test set includes the El Niño seasons of 2023 and 2026. Live scoring runs from launch onward.
- **Wet-season caveat:** launching in November 2026 means the live scorecard stays quiet until the dry season of about June–October 2027. The 2023 and 2026 test seasons carry the evaluation story until then.

## Data sources (all free)

| Layer | Source | Notes |
|---|---|---|
| Active fires (target and fire-history inputs) | NASA FIRMS VIIRS 375 m: yearly country archives for training, area API (free MAP_KEY) for live data | Prefer one consistent sensor (S-NPP, with NOAA-20 filling gaps). Check what's available and document the choice. |
| Weather, training | ERA5 hourly → daily, 0.25°, via Copernicus CDS (`cdsapi`) | Variables: temperature, dewpoint→RH, wind u/v, precipitation, top-layer soil moisture |
| Weather, live | ECMWF open data IFS 0.25° (`ecmwf-opendata`) | The same model family as ERA5, which limits differences between training and live data |
| Fire danger | Fire Weather Index codes (FFMC, DMC, DC, ISI, BUI, FWI) computed with `xclim` from those same weather variables | Computed by one pipeline for both training and live data |
| Rainfall memory | Rolling 7/30/60/90-day rainfall totals and days since rain >1 mm | Built from the weather data |
| Peatland | Global Forest Watch "Indonesia peat lands" map | Static: fraction of each cell |
| Recent clearing | Hansen Global Forest Change `lossyear` (30 m) | Fraction cleared in the 1–3 years *before* the target year (no leakage) |
| Land cover | C3S/ESA CCI yearly land cover, 300 m | Class fractions per cell, using the previous year's map |
| Elevation | Copernicus DEM GLO-90 | Static |
| El Niño state | NOAA Oceanic Niño Index (ONI), monthly | Lagged one month so it was known at forecast time |
| Seasonality | Day of year encoded as sine/cosine | |
| Province boundaries | geoBoundaries ADM1/ADM2 for Indonesia | Used for the per-province summaries in the app |

**Live weather state:** the job keeps the last 90 days of weather on the Hugging Face Hub, starts it once from ERA5, and appends IFS analyses each day. v1 does **not** use future weather forecasts, so training and live inputs match. Adding forecast weather is a v2 experiment.

## Modeling

1. **Baselines** (`models/baselines.py`):
   - persistence (fire today means fire tomorrow)
   - climatology (each cell's historical fire rate for that month, from training years only)
   - logistic regression
   - **LightGBM** on per-cell features plus neighborhood aggregates
2. **Deep learning model** (`models/unet.py`): a hand-written compact U-Net with 4 levels and 32 base channels. The input stacks about 80 channels: 14 days of fire counts and fire intensity (FRP), 7 days of daily weather, the rolling rainfall sums, the fire-weather codes, the static map layers, seasonality and ONI. The output is 7 sigmoid maps, one per lead day.
   - Loss: focal or weighted BCE, because fire cells are rare.
   - Training: random 64×64 crops for augmentation, mixed precision. This fits easily in 4 GB of VRAM.
3. **Experiments** to report:
   - how much each input group adds (ablation)
   - stacking past days as channels vs a ConvLSTM time encoder
   - loss choice
   - whether the deep learning model beats LightGBM overall and on new ignitions
4. **Metrics** (`models/evaluate.py`):
   - PR-AUC per lead day
   - Brier score and calibration curves
   - **top-5% coverage**: the share of fires that fall in the 5% of cells ranked riskiest, which mirrors where patrol teams would go
   - every metric sliced by new ignitions, peat vs non-peat, El Niño vs normal years, and province
5. **Explaining the predictions:** permutation importance per input group, and integrated gradients (Captum) for "why is this area high-risk today?"

## Live system

- **Daily job** (`.github/workflows/daily.yml`, CPU only, about 10 minutes):
  1. fetch the latest FIRMS detections and IFS weather
  2. update the rolling state
  3. run the model
  4. export results
  5. score past forecasts whose outcomes are now known
- **Storage:** a Hugging Face Hub dataset repo holds the rolling state, the forecast archive (parquet) and the scorecard. A model repo holds the weights. Recruiters can download both. GitHub secrets: `FIRMS_MAP_KEY`, `HF_TOKEN`, `CDSAPI_KEY`.
- **Exports per day:**
  - 7 transparent PNG risk overlays, one per lead day
  - `forecast.json` with metadata, a per-province risk summary and the highest-risk cells
  - yesterday's observed fire points
  - `scorecard.json` with rolling metrics vs persistence
- **Web app** (`web/`, Vite + TypeScript + MapLibre GL, static site deployed on Vercel). It never sleeps, unlike free Streamlit or Hugging Face Spaces apps that take a while to wake up.
  - map with a lead-day slider, observed-fire toggle and province panel
  - scorecard page: model vs persistence over time
  - "about the model" page that links to the write-up

## Repo layout

```
pyproject.toml            # uv; torch (CUDA 12), xarray, zarr, rioxarray, geopandas, cdsapi,
                          # ecmwf-opendata, eccodes/cfgrib, xclim, lightgbm, captum, huggingface_hub
src/titikapi/
  config.py  grid.py      # region, grid, land mask, point→cell rasterization
  data/  firms.py era5.py ifs_live.py static.py enso.py features.py   # → local Zarr data cube
  models/ baselines.py dataset.py unet.py train.py evaluate.py
  serve/  daily.py export.py score.py hub.py
notebooks/ 01_eda.ipynb 02_baselines.ipynb 03_results.ipynb
tests/                    # grid, leakage, metrics, FWI sanity, pipeline smoke test
web/                      # Vite + MapLibre
docs/design.md            # this design, committed
README.md                 # write-up: problem, data, results tables/plots, limitations, live link
```

The data cube for 2012–2026 is roughly 1–2 GB of compressed Zarr in `data/`, which is gitignored.

## Milestones

1. **Setup and data cube:** run git init, create the uv project, commit the design doc. Build the FIRMS, ERA5, fire-weather and static layers into a Zarr cube, then do EDA covering fire seasonality, El Niño years and peat.
2. **Baselines and evaluation framework:** persistence, climatology and LightGBM, with the full metric suite and slices.
3. **Deep learning model:** U-Net training, experiments, explanations, and a results notebook.
4. **Daily pipeline:** IFS and FIRMS ingestion, Hugging Face Hub storage, scoring, GitHub Action.
5. **Web app and write-up:** MapLibre site on Vercel, then the README story with the main charts.
6. **Stretch goals:** smoke drift arrows from wind forecasts (the haze part), forecast-weather inputs, coverage of all of Indonesia.

## Accounts the user will need (free)

- NASA FIRMS MAP_KEY
- Copernicus CDS account, with the ERA5 licence accepted
- Hugging Face account and write token
- GitHub repo
- Vercel account

## Verification

- `uv run pytest`, covering:
  - point→cell rasterization against hand-computed cells
  - **leakage tests:** features at day t use only data from day t or earlier, and climatology and forest loss use only earlier years
  - metric functions checked against sklearn
  - FWI values in sensible ranges for a known dry spell, such as Sept 2019 in Riau
- Build the cube for one month and visually check fire counts against the FIRMS web map.
- Check that the baseline metrics table reproduces, and that persistence beats random (a sanity check).
- Training: validation PR-AUC beats LightGBM on new ignitions. If it doesn't, report that honestly.
- `uv run python -m titikapi.serve.daily --date 2025-09-15 --dry-run` produces the PNGs and JSON for a past date. Compare them with the fires actually observed.
- Trigger the GitHub Action manually and confirm the files appear on the Hugging Face Hub.
- Run `npm run dev` in `web/` and check that the map loads, the slider switches lead days and the scorecard renders. Then deploy and open the public URL.
