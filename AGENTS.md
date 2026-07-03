# AGENTS.md

Context for Codex working in this repo.

## Project goal

Predict MLB game outcomes (binary home-team-wins and total runs) from free data sources. Optimize for calibration and log loss, not just accuracy. The model should be retrainable nightly on a laptop in under 30 minutes.

## Data flow

```
External APIs ──► data/raw/        (cached, immutable, parquet)
                       │
                       ▼
                 data/processed/    (cleaned game logs, joined IDs)
                       │
                       ▼
                 src/features/      (rolling windows, joins → model input)
                       │
                       ▼
                 src/models/        (train, evaluate, predict)
```

**Never** fetch raw data from inside `src/features/` or `src/models/`. Ingest is a separate stage and its outputs are cached on disk.

## Package conventions

- Python 3.11+. Use type hints on all public functions.
- Use `pandas` for tabular data, `pyarrow`/parquet for storage. CSV only for human inspection.
- Use `pybaseball` for historical pulls; cache aggressively in `data/raw/` keyed by `(source, start_date, end_date)`.
- Use the `MLB-StatsAPI` package (or raw `requests` to `statsapi.mlb.com`) for current/upcoming games.
- Logging via stdlib `logging`, not `print`. Configure once in `src/utils/logging.py`.
- Configuration in `.env` (loaded by `python-dotenv`); never hardcode paths.

## Model modes & feature eligibility

The model trains and serves in one of two modes (`src/models/feature_config.py`):

- **`pregame_safe` (default)** — uses only the **126 features** a leakage audit proved are available before first pitch and free of target leakage. This is what `scripts/train_model.py` and the backend use by default. Artifacts: `models/{home_win,total_runs}_pregame_safe_latest.pkl`.
- **`legacy_full`** — the full **169-feature** set (`FEATURE_COLS` in `train.py`), kept for comparison only. Artifacts: `models/{home_win,total_runs}_latest.pkl`.

The pregame-safe list lives in `data/processed/leakage_safe_feature_cols.json`, generated from `diagnostics/feature_inventory.csv`. In `pregame_safe` mode, training and prediction use exactly that list, and `src/models/predict.py` runs an as-of timestamp check that raises if any feature could leak. See `diagnostics/feature_leakage_audit.md` and `diagnostics/feature_eligibility_report.md`.

## Feature groups (169-feature inventory; 126 pregame-safe)

| Group | Count | Status | Source |
|---|---|---|---|
| Team rolling form (L5/L10/L20) | 30 | safe | `data/processed/games/` |
| SP traditional stats (rolling L3 + season-to-date) | 28 | safe | `data/raw/pitching_gamelogs/` |
| Schedule / rest (team + SP days rest) | 4 | safe | `data/processed/games/` |
| SP availability / sample-size flags | 8 | safe | `data/processed/games/` + live feed |
| Park factors (pf_runs, pf_hr) | 2 | safe | FanGraphs / game-log fallback |
| Season-to-date team rates + home/away splits | 6 | safe | `data/processed/games/` |
| SP Statcast quality (xwOBA against, whiff, barrel, platoon) | 14 | safe | `data/raw/statcast/` |
| Team batting Statcast (xwOBA off, barrel rate, rolling L10) | 4 | safe | `data/raw/statcast/` |
| Bullpen quality (L14) + workload/fatigue (L1–3d) | 22 | safe | `data/raw/statcast/` |
| Missingness indicators (safe-feature subset) | 8 | safe | derived |
| Posted lineup Statcast (with team fallback) | 13 | **excluded** — possible leakage | `data/raw/lineups/`, `data/raw/statcast/` |
| Lineup matchup (xwOBA vs SP hand + BvP) | 4 | **excluded** — definite leakage | `data/raw/statcast/` |
| SP/BP pitch-arsenal quality | 16 | **excluded** — definite leakage | Baseball Savant |
| Missingness indicators (excluded-feature subset) | 10 | excluded | derived |

**Why the exclusions:** lineup-matchup/BvP features were built from *same-game* Statcast plate appearances; pitch-arsenal features applied a season-level snapshot with no as-of cutoff; posted-lineup aggregates can't yet prove pre-first-pitch availability. Re-enabling any of these requires a true as-of snapshot captured before first pitch.

## Where things go

- New team/schedule feature → `src/features/team.py` or `src/features/build.py:build_training_set()`. Function takes long-format `team_games` DataFrame, returns per `(game_pk, team_id)`.
- New pitcher feature → `src/features/pitcher.py` (single-pitcher helpers) or `src/features/build.py:join_gamelog_pitcher_features()` / `join_pitcher_features()` (pipeline joins).
- New batting / lineup feature → `src/features/batting.py` (per-lineup helpers) or `src/features/build.py:join_lineup_matchup_features()`.
- New Statcast aggregation → `src/data/statcast.py`. Must be vectorised (groupby, not Python loops).
- New model → sibling in `src/models/`. Always save with versioned filename: `models/{model_name}_{YYYYMMDD}.pkl` plus a `models/{model_name}_latest.pkl`.
- New data source → new module in `src/data/`. Must implement a `fetch(start, end) -> DataFrame` and write to `data/raw/{source}/`.
- New script → `scripts/`. Scripts are thin: parse args, call `src/`, print/save.
- Add new feature columns to `FEATURE_COLS` in `src/models/train.py`, classify them in `diagnostics/feature_inventory.csv` (`leakage_risk` + `safe_before_first_pitch`) so they can enter `pregame_safe`, AND add them to `_FACTOR_COLS` in `backend/services/assemble.py` so the dashboard reflects them. A feature missing from the inventory never reaches the default model.

## Don't do this

- Don't commit anything under `data/` (already in `.gitignore`). Raw pulls can be GBs.
- Don't fetch data inside feature engineering or model code.
- Don't introduce target leakage. Rolling windows must use only games strictly **before** the target game's date. When in doubt, write a test that confirms no row uses same-day or future data.
- Don't add a feature to `pregame_safe` without an as-of timestamp proving it's known before first pitch. Same-game and season-snapshot sources leak; classify honestly in `diagnostics/feature_inventory.csv` (this is exactly how lineup-matchup and pitch-arsenal features got excluded).
- Don't use scikit-learn's default cross-validation for time series — use `TimeSeriesSplit` or a custom walk-forward backtest. Random k-fold leaks the future into the past.
- Don't drop NaNs silently. Pitcher rolling stats will be NaN for early-career starts; decide explicitly (impute league avg, mark as rookie flag, etc.).
- Don't rely on team abbreviations as join keys — they collide and change. Use MLBAM team IDs.

## Backtesting

Always evaluate on held-out seasons, not random splits. Canonical split: **train 2018–2023, validate 2024, test 2025** (the `train_model.py` defaults). Update as more data accumulates.

Track per run: log loss, Brier score, calibration plot, and accuracy at the 0.5 threshold. The bar is beating "always pick home team" by a meaningful log-loss margin.

**Current benchmark — leakage-safe comparison** (`diagnostics/backtests/leakage_safe_comparison/`, generated 2026-05-27; train 2018–2022, early-stop val 2023, eval 2024–2025, 4,729 games, isotonic calibration fit on 2023):

| Model | Accuracy | Brier | Log loss | High-conf acc |
|---|---|---|---|---|
| `legacy_full` (169 feat) | 0.557 | 0.242 | 0.677 | 0.649 (1,372 g) |
| `pregame_safe` (126 feat, **default**) | 0.552 | 0.244 | 0.688 | 0.615 (1,447 g) |
| best simple baseline (run-diff) | 0.538 | 0.249 | 0.691 | — |
| home-team baseline | 0.526 | 0.249 | 0.692 | — |

Removing the leaky features costs ~0.5 pts accuracy / +0.011 log loss. `pregame_safe` still edges the best simple baseline on log loss but is the honest, deployable number — and it remains mildly overconfident (avg winner confidence 0.563 vs 0.552 accuracy), so keep the calibration layer. The total-runs regressor trains in the same two modes; check `train_model.py` output for current MAE. Regenerate this table with `python scripts/backtest_leakage_safe_comparison.py`.

## Testing

`pytest` from repo root. Tests that hit external APIs go in `tests/integration/` and are skipped by default (`pytest -m integration` to run). Unit tests for feature functions should use small fixture DataFrames in `tests/fixtures/`.

## Common commands

```bash
pip install -e ".[dev]"                        # editable install with dev deps
pytest                                          # unit tests
pytest -m integration                           # API-hitting tests
ruff check src/ tests/                          # lint
ruff format src/ tests/                         # format

python scripts/fetch_data.py                    # refresh schedule/score cache
python scripts/fetch_pitching_gamelogs.py       # refresh SP gamelog cache
python scripts/fetch_statcast.py                # refresh Statcast pitch cache

python scripts/train_model.py                   # train both models (pregame_safe by default)
python scripts/train_model.py --model-mode legacy_full   # train the 169-feature comparison model
python scripts/tune_hyperparams.py              # grid-search LightGBM params
python scripts/predict_today.py                 # today's slate predictions
python scripts/backtest_leakage_safe_comparison.py  # pregame_safe vs legacy vs baselines
python scripts/build_static_slate.py --date 2026-06-24  # write public/slates/{date}.json for the React dashboard
python scripts/build_dashboard_data.py --team 147  # regenerate Yankees dashboard JSON
python scripts/feature_importance.py            # ranked feature importance report
```

## Dashboard serving

The React/Vite dashboard (root `src/App.tsx`, "Diamond Forecast") reads predictions from **static JSON at `public/slates/{date}.json`** (written by `scripts/build_static_slate.py`) plus the live MLB schedule API — it does **not** call the FastAPI backend, and falls back to mock data in `src/data/mockModelData.ts` when no slate exists. The FastAPI backend (`backend/`, `/api/predictions/*`) is a separate, richer live-pipeline serving path. Note: the Python package and the React app both live under `src/`.
