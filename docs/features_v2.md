# Feature V2 Notes

## Posted Lineups

Source: MLB Stats API live feed (`/api/v1.1/game/{gamePk}/feed/live`), parsed in `src/data/lineups.py`.

Cache: one parquet row per game at `data/raw/lineups/{game_pk}.parquet`.

Refresh cadence: lineups can change before first pitch, so current-day prediction runs should refresh or invalidate the per-game cache when the lineup card is still not confirmed. Historical backfills use post-game boxscores and treat available nine-player batting orders as confirmed.

Features:

- `home_lineup_xwoba_vs_hand_L30`, `away_lineup_xwoba_vs_hand_L30`
- `home_lineup_xwoba_weighted`, `away_lineup_xwoba_weighted`
- `home_lineup_xwoba_top5`, `away_lineup_xwoba_top5`
- `home_lineup_barrel_rate_vs_hand_L30`, `away_lineup_barrel_rate_vs_hand_L30`
- `home_lineup_barrel_rate_weighted`, `away_lineup_barrel_rate_weighted`
- `home_lineup_barrel_rate_top5`, `away_lineup_barrel_rate_top5`
- `lineup_features_missing`

Computation: each batter's prior-30-day, prior-100-PA Statcast xwOBA and barrel rate are filtered to the opposing starter's handedness. Lineup aggregates include a simple mean, a slot-weighted mean, and a top-five mean. Slot weights are `1.20/1.15/1.15/1.10/1.05/1.00/0.95/0.90/0.85`.

Gotcha: `confirmed` means a batting order is available from the official feed. `projected` is reserved for official pre-game placeholder lineups. `missing` means the API did not provide a usable nine-player order. When status is not confirmed, the model falls back to the existing team rolling Statcast offense features and sets `lineup_features_missing = 1`.

## Baseball Savant Pitch Arsenal Quality

Source: Baseball Savant Pitch Arsenal Stats leaderboard in `src/data/pitch_quality.py`.

Endpoint: `https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats?type=pitcher&season={season}&min={min_pitches}&csv=true`

Cache:

- Season tables: `data/raw/pitch_quality/pitch_arsenal_{season}_{as_of_or_latest}.parquet`

Refresh cadence: current-season latest tables are treated as fresh for 24 hours. Historical season caches are immutable unless force-refetched.

Starter features:

- `home_sp_rv_per_100`, `away_sp_rv_per_100`
- `home_sp_xwoba_arsenal`, `away_sp_xwoba_arsenal`
- `home_sp_whiff_arsenal`, `away_sp_whiff_arsenal`
- `home_sp_pitch_quality_missing`, `away_sp_pitch_quality_missing`

Bullpen features:

- `home_bp_rv_per_100_weighted`, `away_bp_rv_per_100_weighted`
- `home_bp_xwoba_arsenal_weighted`, `away_bp_xwoba_arsenal_weighted`
- `home_bp_whiff_arsenal_weighted`, `away_bp_whiff_arsenal_weighted`
- `home_bp_pitch_quality_missing`, `away_bp_pitch_quality_missing`

Computation: Savant returns one row per pitcher/pitch type. The module aggregates to pitcher level by weighting `run_value_per_100`, `est_woba`, and `whiff_percent` by `pitch_usage` when available, falling back to pitch counts. Starter features join directly on MLBAM `player_id`. Bullpen features use a weighted average of the top five team pitchers by pitch count in the cached season table.

Gotcha: `rv_per_100` is an empirical run-value measure, not a Stuff+ model. We are using it as a public Stuff+ substitute because FanGraphs now blocks scraper access. Smaller samples can be noisy, and players below the configured minimum pitch threshold will get the existing pitch-quality missingness flags.
