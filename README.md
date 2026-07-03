# mlb-predictor

Machine-learning models that predict MLB game outcomes — home-team win probability
and total runs — using only **free, keyless data sources**. The project optimizes for
calibration and log loss (not just accuracy) and is designed to retrain nightly on a
laptop in under 30 minutes.

## Model modes

The model trains and serves in one of two feature configurations
(`src/models/feature_config.py`):

- **`pregame_safe` (default)** — the **126 features** a leakage audit proved are
  available before first pitch and free of target leakage. This is what
  `scripts/train_model.py`, `scripts/predict_today.py`, and the backend use by default.
  Artifacts: `models/{home_win,total_runs}_pregame_safe_latest.pkl`.
- **`legacy_full`** — the full **169-feature** set, kept for comparison only.
  Artifacts: `models/{home_win,total_runs}_latest.pkl`.

The pregame-safe list lives in `data/processed/leakage_safe_feature_cols.json`, generated
from `diagnostics/feature_inventory.csv`. In `pregame_safe` mode, prediction runs an
as-of timestamp check that raises if any feature could leak. See
`diagnostics/feature_leakage_audit.md` and `CLAUDE.md` for the full audit.

## Data sources

- **[pybaseball](https://github.com/jldbc/pybaseball)** — wraps Baseball Reference,
  FanGraphs, and Baseball Savant (Statcast). Used for historical training data.
- **[MLB Stats API](https://statsapi.mlb.com)** — keyless public API. Used for the daily
  slate, probable starters, box scores, and live game state.

All sources are free and require no API key.

## Predictors (current, pregame-safe)

- Team rolling form (wins, run differential, runs for/against) over L5 / L10 / L20
- Season-to-date team rates and home/away splits
- Starting-pitcher rolling stats (ERA, WHIP, K/9, BB/9, HR/9, IP) — last 3 starts and
  season-to-date
- Starting-pitcher Statcast quality (xwOBA against, whiff, barrel, platoon splits)
- Team batting Statcast (offensive xwOBA, barrel rate, rolling L10)
- Bullpen quality (L14) plus recent workload / fatigue (L1–3 days)
- Park factors (prior-year `pf_runs`, `pf_hr`)
- Schedule / rest (team and starter days rest) and starter availability flags

Some richer families (posted-lineup aggregates, lineup-vs-starter / batter-vs-pitcher
matchup, and pitch-arsenal quality) are implemented but **excluded from the default
model** because they can't yet prove they're known before first pitch. See the exclusion
table in `CLAUDE.md`.

> **Not yet implemented:** weather and umpire features. They are on the roadmap but are
> not part of any current model; ignore any leftover placeholder references in the UI.

## Quickstart

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# 1) Build the cached data (slow on first run; cached to data/ afterwards)
python scripts/fetch_data.py --start 2018 --end 2025   # schedule + scores
python scripts/fetch_pitching_gamelogs.py              # starter game logs
python scripts/fetch_statcast.py                       # Statcast pitch cache

# 2) Train both models (pregame_safe by default)
python scripts/train_model.py
python scripts/train_model.py --model-mode legacy_full # 169-feature comparison model

# 3) Predict today's slate
python scripts/predict_today.py

# 4) Honest backtest: pregame_safe vs legacy vs simple baselines
python scripts/backtest_leakage_safe_comparison.py
```

Configuration is read from `.env` (copy `.env.example` → `.env`); nothing is hardcoded.
Data and model artifacts are written under `data/` and `models/`, both gitignored.

## Backtesting

Evaluation is always on held-out **seasons**, never random splits. The canonical
comparison (`diagnostics/backtests/leakage_safe_comparison/`) trains on 2018–2022,
early-stops on 2023, and evaluates on 2024–2025 with isotonic calibration fit on 2023:

| Model | Accuracy | Brier | Log loss |
|---|---|---|---|
| `legacy_full` (169 feat) | 0.557 | 0.242 | 0.677 |
| `pregame_safe` (126 feat, **default**) | 0.552 | 0.244 | 0.688 |
| best simple baseline (run-diff) | 0.538 | 0.249 | 0.691 |
| home-team baseline | 0.526 | 0.249 | 0.692 |

`pregame_safe` is the honest, deployable number and still edges the best simple baseline
on log loss. Regenerate with `python scripts/backtest_leakage_safe_comparison.py`.

## Dashboard (Diamond Forecast)

The React/Vite front-end (`src/App.tsx`) reads predictions from **static JSON at
`public/slates/{date}.json`** (written by `scripts/build_static_slate.py`) plus the live
MLB schedule API. It falls back to mock data in `src/data/mockModelData.ts` when no slate
exists — so it runs with no backend.

```bash
npm install
npm run dev                                       # Vite dev server
python scripts/build_static_slate.py --date 2026-06-24   # write public/slates/{date}.json
```

A separate FastAPI backend (`backend/`, `uvicorn backend.main:app`) exposes a richer
live-prediction API (`/api/predictions/*`); it is independent of the static-slate
dashboard path.

## Testing & lint

```bash
pytest                        # unit tests (API-hitting tests are skipped by default)
pytest -m integration         # tests that hit external APIs
ruff check src/ tests/        # lint
ruff format src/ tests/       # format
```

## Layout

```
src/         importable library code (data ingest, features, models) AND the React app
scripts/     CLI entry points (the things you actually run)
backend/     FastAPI live-prediction API (separate from the static dashboard)
data/        raw/ -> processed/ -> features for training; not committed
models/      saved model artifacts; not committed
diagnostics/ leakage audit, feature inventory, and backtest reports
tests/       pytest
```

Both the Python package and the React app live under `src/`. See `CLAUDE.md` for the full
conventions and the leakage audit.
