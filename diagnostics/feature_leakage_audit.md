# Feature Leakage Audit

Generated: 2026-05-25
Model feature count audited: 169

## Executive summary

The model is not uniformly unsafe. Most team, starter-gamelog, starter-Statcast, team-batting, bullpen, rest, and park-factor features are built from shifted or strictly prior-date data and are classified as Safe.

The high-risk areas are concentrated in two feature families:

1. Lineup matchup features are Definite leakage in the training path because `join_lineup_matchup_features` derives effective lineups from same-game Statcast plate appearances and derives starter handedness from same-game pitches.
2. Pitch-quality features are Definite leakage in the training path because `join_pitch_quality_features` applies one season/latest Baseball Savant pitch-arsenal snapshot to every game in that season, without an as-of cutoff.

Posted-lineup aggregate features are Possible leakage. Live prediction can use confirmed pregame lineups, but historical training uses `historical=True` lineups from live feed/boxscore data after the fact, so the audit cannot prove those values were available before first pitch.

## Classification counts

| Leakage risk | Feature count |
|---|---:|
| Safe | 126 |
| Possible leakage | 15 |
| Definite leakage | 28 |
| Unknown | 0 |

## Definite leakage features

`home_lineup_xwoba_vs_sp`, `away_lineup_xwoba_vs_sp`, `home_bvp_xwoba`, `away_bvp_xwoba`, `home_sp_rv_per_100`, `home_sp_xwoba_arsenal`, `home_sp_whiff_arsenal`, `away_sp_rv_per_100`, `away_sp_xwoba_arsenal`, `away_sp_whiff_arsenal`, `home_bp_rv_per_100_weighted`, `home_bp_xwoba_arsenal_weighted`, `home_bp_whiff_arsenal_weighted`, `away_bp_rv_per_100_weighted`, `away_bp_xwoba_arsenal_weighted`, `away_bp_whiff_arsenal_weighted`, `home_sp_pitch_quality_missing`, `away_sp_pitch_quality_missing`, `home_bp_pitch_quality_missing`, `away_bp_pitch_quality_missing`, `home_lineup_xwoba_vs_sp_missing`, `away_lineup_xwoba_vs_sp_missing`, `home_bvp_xwoba_missing`, `away_bvp_xwoba_missing`, `home_sp_rv_per_100_missing`, `away_sp_rv_per_100_missing`, `home_bp_rv_per_100_weighted_missing`, `away_bp_rv_per_100_weighted_missing`

## Possible leakage features

`home_lineup_xwoba_vs_hand_L30`, `home_lineup_xwoba_weighted`, `home_lineup_xwoba_top5`, `home_lineup_barrel_rate_vs_hand_L30`, `home_lineup_barrel_rate_weighted`, `home_lineup_barrel_rate_top5`, `away_lineup_xwoba_vs_hand_L30`, `away_lineup_xwoba_weighted`, `away_lineup_xwoba_top5`, `away_lineup_barrel_rate_vs_hand_L30`, `away_lineup_barrel_rate_weighted`, `away_lineup_barrel_rate_top5`, `lineup_features_missing`, `home_lineup_xwoba_vs_hand_L30_missing`, `away_lineup_xwoba_vs_hand_L30_missing`

## Safe feature families

| Family | Evidence | Notes |
|---|---|---|
| Team rolling form | `compute_team_rolling_features` uses `shift(1)`; `_team_snapshot` filters `game_date < target_date`. | Safe for target game; conservative for doubleheaders because same-date completed game 1 is excluded from game 2 predictions. |
| Team season-to-date rates and splits | `season_to_date` uses shifted expanding sums; `home_away_split` emits rates before updating with the current row. | Safe, subject to processed cache freshness diagnostics. |
| Starter gamelog stats | `compute_starter_rolling_features` and `compute_starter_season_to_date` use `shift(1)`; live path filters prior starts. | Safe. Missing probable pitchers are represented through unknown/missing flags. |
| Starter Statcast rolling quality | `compute_pitcher_rolling_features` shifts prior starts; live path loads through target_date - 1 day. | Safe, with a recommended max source date assertion. |
| Team batting Statcast | Training rolls with shifted per-game team aggregates; live path uses prior-day history. | Safe. |
| Bullpen quality and workload | Rolling quality and workload are computed from prior games; live path uses prior-day data. | Safe, conservative for doubleheaders. |
| Park factors | `_join_park_factors` documents and uses Y-1 factors for games in year Y. | Safe. |
| Train/test split | `time_based_split` uses chronological date ranges, not random split. | Safe for temporal validation. |

## Leakage findings

### 1. Lineup matchup training features use same-game Statcast data

`src/features/build.py::join_lineup_matchup_features` calls `aggregate_game_lineups(pitches)`, and `src/data/statcast.py::aggregate_game_lineups` builds lineups from batters who actually appeared in the game. The same join also gets starter handedness from `aggregate_pitcher_starts(pitches)` for that game. Those facts are not guaranteed before first pitch and are produced by the target game itself.

Impact: `home_lineup_xwoba_vs_sp`, `away_lineup_xwoba_vs_sp`, `home_bvp_xwoba`, `away_bvp_xwoba`, and their missingness flags are marked Definite leakage.

Recommended fix: train these features only from immutable pregame lineup snapshots and probable starter metadata captured before first pitch. Until then, remove these features from training or split into a clearly labeled post-lineup model that only runs after verified lineup snapshot time.

### 2. Pitch-quality features use season/latest snapshots without as-of dates

`src/features/build.py::join_pitch_quality_features` calls `fetch_pitch_quality(season)` once per season and applies that data to every game in the season. `src/data/pitch_quality.py::fetch_pitch_quality` accepts `as_of`, but the join does not pass it, and the fetch URL is season-level. Historical early-season rows can therefore see future season pitch-arsenal quality.

Impact: all starter and bullpen pitch-quality features and their missingness flags are marked Definite leakage.

Recommended fix: cache pitch-quality snapshots by `as_of` date and join the most recent snapshot strictly before first pitch. A simpler conservative alternative is to use prior-season pitch-quality features only.

### 3. Historical posted lineup aggregates have unproven pregame availability

`join_posted_lineup_features` loads historical lineups with `historical=True`, then treats confirmed lineups as training input. In live prediction, `build_posted_lineup_prediction_features` uses `historical=False` and allows missing lineups, which is closer to a pregame workflow. The training source still lacks fetch timestamp and availability proof.

Impact: posted lineup aggregate features are marked Possible leakage rather than Safe.

Recommended fix: persist lineup snapshots with `fetched_at`, `game_start_time`, source status, and raw feed status. Train only on snapshots whose `fetched_at < first_pitch`.

## Freshness and dashboard observations

- No odds or market-line feature is present in `FEATURE_COLS`. The frontend mock data mentions market line, but that is not a model input in this audit.
- Standings and hitting logs are fetched in dashboard assembly for display, reasons, and preview fallback. The full model feature path uses processed historical games and explicit feature builders, not dashboard standings as model input.
- The dashboard preview fallback should stay clearly separate from model predictions because live standings can include games completed earlier that day.
- Result merging in `build_training_set` adds targets after feature joins. The target columns are not in `FEATURE_COLS`; leakage risk comes from feature source timing, not direct target-column reuse.
- The train/validation/test split is chronological via `time_based_split`; no random train/test split was found in the active model training code.

## Recommended next audit/fix order

1. Remove or gate Definite leakage features from training until pregame/as-of sources exist.
2. Add source freshness columns to prediction audit rows: max processed game date, max Statcast date, lineup fetched_at, pitch-quality as_of.
3. Add a training-set assertion that every feature family declares an effective timestamp less than first pitch.
4. Add tests for lineups and pitch-quality as-of joins before re-enabling those feature families.

See `diagnostics/feature_inventory.csv` for the feature-by-feature inventory.
