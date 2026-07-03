# Leakage-Safe Model Comparison Backtest

Generated: 2026-05-27

## Setup

- Training window: 2018 through 2022-12-31
- Early-stopping validation window: 2023 through 2023-12-31
- Evaluation window: 2024-01-01 through 2025-12-31
- Split type: chronological only
- Odds: not used
- Pregame-safe probabilities: isotonic calibration fitted on the 2023 validation slice
- Evaluation games: 4729
- Pregame-safe feature coverage in this local matrix: 126 of 126 audited safe features
- Full all-feature historical matrix was used, with all audited safe features materialized.

## Model Comparison

| model_name | accuracy | brier_score | log_loss | high_confidence_accuracy | high_confidence_games | games_predicted | games_excluded | avg_winner_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_full | 0.5572 | 0.2423 | 0.6773 | 0.6494 | 1372 | 4729 | 0 | 0.5751 |
| pregame_safe | 0.5517 | 0.2441 | 0.6881 | 0.6151 | 1447 | 4729 | 0 | 0.5632 |
| better_run_differential_baseline | 0.5384 | 0.2490 | 0.6911 | nan | 0 | 4729 | 0 | 0.5590 |
| better_record_baseline | 0.5413 | 0.2490 | 0.6911 | nan | 0 | 4729 | 0 | 0.5670 |
| home_team_baseline | 0.5259 | 0.2494 | 0.6919 | nan | 0 | 4729 | 0 | 0.5200 |
| better_last_10_baseline | 0.5291 | 0.2495 | 0.6922 | nan | 0 | 4729 | 0 | 0.5488 |
| elo_baseline | 0.5468 | 0.2531 | 0.7012 | 0.5748 | 2587 | 4729 | 0 | 0.6248 |

## Explicit Answers

1. How much did performance drop after removing leakage?

Accuracy dropped by 0.55 percentage points for `pregame_safe` versus `legacy_full`. Brier changed by 0.0018, and log loss changed by 0.0108. Lower Brier/log loss is better.

2. Does pregame_safe beat the simple baselines?

`pregame_safe` beats the best simple baseline by log loss. Best baseline by log loss is `better_run_differential_baseline` with log loss 0.6911; `pregame_safe` log loss is 0.6881.

3. Is the model still overconfident?

`pregame_safe` is still materially overconfident in at least one confidence check. Average winner confidence is 0.5632 versus accuracy 0.5517, a gap of 0.0115.

4. Which segments perform worst?

Worst `pregame_safe` months by log loss:

| model_name | month | games | accuracy | brier_score | log_loss | avg_winner_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| pregame_safe | 2024-07 | 364 | 0.5137 | 0.2507 | 0.7268 | 0.5616 |
| pregame_safe | 2025-06 | 397 | 0.5189 | 0.2474 | 0.7170 | 0.5612 |
| pregame_safe | 2024-09 | 388 | 0.5696 | 0.2414 | 0.7052 | 0.5702 |
| pregame_safe | 2025-07 | 361 | 0.5152 | 0.2517 | 0.6970 | 0.5534 |
| pregame_safe | 2024-05 | 413 | 0.5400 | 0.2479 | 0.6898 | 0.5646 |

Worst `pregame_safe` teams by accuracy when they appear:

| model_name | team_id | games | accuracy_when_in_game | avg_model_p_team_win | model_pick_rate | actual_team_win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| pregame_safe | 139 | 315 | 0.4857 | 0.4907 | 0.4698 | 0.4825 |
| pregame_safe | 138 | 315 | 0.5111 | 0.4807 | 0.3810 | 0.4952 |
| pregame_safe | 140 | 315 | 0.5111 | 0.4971 | 0.5048 | 0.4857 |
| pregame_safe | 114 | 316 | 0.5158 | 0.4898 | 0.4462 | 0.5570 |
| pregame_safe | 134 | 315 | 0.5206 | 0.4860 | 0.4159 | 0.4508 |
| pregame_safe | 135 | 312 | 0.5224 | 0.5246 | 0.5994 | 0.5641 |
| pregame_safe | 141 | 315 | 0.5238 | 0.5015 | 0.5175 | 0.5238 |
| pregame_safe | 113 | 316 | 0.5253 | 0.4891 | 0.4272 | 0.4905 |

Probable-pitcher availability:

| model_name | pitcher_availability | games | accuracy | brier_score | log_loss | avg_winner_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| pregame_safe | both_probable_pitchers_known | 4717 | 0.5514 | 0.2441 | 0.6882 | 0.5632 |
| pregame_safe | one_or_more_probable_pitchers_missing | 12 | 0.6667 | 0.2278 | 0.6482 | 0.5536 |

5. What should be fixed next?

- Keep `pregame_safe` as the default candidate for deployable pregame evaluation.
- Add timestamped first-pitch source freshness to the feature matrix so doubleheaders and late lineup changes can be audited at the game level.
- Replace excluded lineup matchup and pitch-quality families with true as-of snapshots before allowing them back into a pregame model.
- Review the worst months/teams above for missing pitcher data, schedule quirks, and systematic team-specific bias.
- Keep the calibration layer in place and monitor it against fresh audited prediction rows before tuning features.

## Output Files

- `model_comparison_metrics.csv`
- `performance_by_confidence_bucket.csv`
- `performance_by_month.csv`
- `performance_by_team.csv`
- `performance_by_pitcher_availability.csv`
- `worst_50_pregame_safe_predictions.csv`
- `excluded_game_reasons.csv`
