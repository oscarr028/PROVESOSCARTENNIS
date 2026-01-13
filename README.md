# Tennis Prediction Pipeline v1 — README

This repo-less bundle contains a single script to build a baseline tennis match win-probability model from historical data, using only **pre-match** information to avoid leakage.

## 1) Expected CSV Inputs

Place files inside a `data/` directory (or pass `--data_dir`):

### `matches.csv` (required)
Columns:
- `match_id` (str, unique)
- `date` (YYYY-MM-DD)
- `tournament` (str)
- `city` (str)
- `country` (str)
- `level` (str: 250/500/1000/GS/Chall/ITF/...)
- `round` (str: R1,R2,QF,SF,F,Q,...)
- `best_of_5` (0/1)
- `surface` (str: hard/clay/grass/indoor-hard)
- `indoor` (0/1)
- `player_a_id` (str)
- `player_b_id` (str)
- `winner_id` (str, equals player_a_id or player_b_id)
- `duration_minutes` (int, optional)
- (optional) `player_a_country`, `player_b_country`

### `points_sets_games.csv` (optional but recommended)
Per (match_id, player_id) aggregates:
- `match_id`, `player_id`
- `aces`, `double_faults`
- `first_sv_in`, `first_sv_pts_won`
- `second_sv_pts_won`, `second_sv_attempts`
- `bp_faced`, `bp_saved`, `bp_opp`, `bp_conv`
- `tb_played`, `tb_won`
- `service_games`, `return_games`

> If you don't have this file, the MVP still runs using Elo + simple form features.

## 2) Outputs

Saved under `outputs/` (or `--out_dir`):
- `features_player_pre.csv` — per-player pre-match features.
- `dataset_match_level.csv` — A–B differential features + target.
- `model_columns.txt` — used feature names.
- `calibration_valid.png` — validation reliability plot.
- `preds_test.csv` — test set predicted probabilities and labels.
- Console printed metrics for train/valid/test.

## 3) Run

```bash
python tennis_model_pipeline_v1.py --data_dir data --out_dir outputs   --train_end 2018-12-31 --valid_end 2025-08-31   --tau_days 60 --alpha_pct 30
```

Adjust dates for your temporal split. The remainder after `valid_end` is held out as **TEST**.

## 4) Notes & Next Steps

- The Elo engine keeps **global** and **surface** ratings, with surface ratings softly blended toward global for stability.
- Rolling/micro stats use **Bayesian shrinkage** to mitigate small-sample noise.
- This skeleton focuses on clarity; for very large datasets, consider vectorized rolling pipelines or incremental parquet features.
- Add more variables (H2H with time decay, ranking snapshots, travel/jetlag, altitude) following the same pattern.
- Consider a **Gradient Boosting** model (LightGBM/XGBoost) as a second baseline and compare via logloss + calibration.
