# NBA Player Stat Predictor

## 1. Project Overview

This project predicts NBA player statistics — Points, Rebounds, and Assists — for any given game day. Predictions are generated before tip-off, compared against Vegas player prop lines, and evaluated after games complete.

**Tech stack:** Python, Ridge Regression, XGBoost, Apache Airflow, Flask, PostgreSQL, Docker

**Current model performance (2025-26 season, full dataset):**

| Stat | MAE  | R2    |
|------|------|-------|
| PTS  | 4.11 | 0.548 |
| REB  | 1.78 | 0.450 |
| AST  | 1.22 | 0.486 |

---

## 2. How It Works - Pipeline Overview

```
NBA API -> Ingestion -> Feature Engineering -> Model -> Predictions -> Dashboard
```

| Step | Description |
|------|-------------|
| **NBA API** | Game logs, rosters, schedules, and scoreboard data pulled via `nba_api` |
| **Ingestion** | Raw player and team game logs saved to `data/raw/` as timestamped CSVs |
| **Feature Engineering** | Rolling averages, rest days, opponent strength, pace, absences, and foul features computed and saved to `data/features/player_features.csv` |
| **Model** | Ridge Regression produces point predictions; XGBoost quantile models produce p10/p90 confidence intervals |
| **Predictions** | Calibration applied, Vegas lines merged, playoff adjustments applied when detected, output saved to `data/predictions/` |
| **Dashboard** | Flask app displays predictions vs Vegas lines with sortable tables, and post-game evaluation results |

---

## 3. Project Structure

```
nba_predictor/
|
|- dags/
|   |- ingest_dag.py          # Airflow: fetch raw game logs at 9am UTC
|   |- features_dag.py        # Airflow: build feature matrix at 10am UTC
|   |- predict_dag.py         # Airflow: generate predictions at 11am UTC
|
|- data/
|   |- raw/                   # Timestamped CSVs of raw game logs
|   |- features/              # player_features.csv (105k+ rows, 157 columns)
|   |- predictions/           # predictions_YYYY-MM-DD.csv, evaluation_YYYY-MM-DD.csv
|
|- models/
|   |- pts_model_ridge.pkl    # Ridge production model (PTS)
|   |- reb_model_ridge.pkl    # Ridge production model (REB)
|   |- ast_model_ridge.pkl    # Ridge production model (AST)
|   |- pts_model_lower.json   # XGBoost p10 quantile model (PTS)
|   |- pts_model_upper.json   # XGBoost p90 quantile model (PTS)
|   |- (same pattern for REB and AST)
|
|- src/
|   |- ingestion/
|   |   |- nba_stats.py       # Fetch player/team game logs via NBA API
|   |   |- roster.py          # Roster and player availability checks
|   |   |- injuries.py        # ESPN injury feed (Out players filtered)
|   |   |- odds.py            # Odds API integration for Vegas prop lines
|   |   |- playoff_stats.py   # Playoff game log fetching and elevation calc
|   |
|   |- features/
|   |   |- engineer.py        # Full feature engineering pipeline
|   |
|   |- models/
|   |   |- train.py           # Train all models (Ridge + XGBoost quantile)
|   |   |- predict.py         # Run daily predictions with all adjustments
|   |   |- evaluate.py        # Post-game evaluation vs actuals
|   |
|   |- dashboard/
|       |- app.py             # Flask application and route logic
|       |- run.py             # Flask entry point
|
|- docker-compose.yml         # Airflow + PostgreSQL services
|- requirements.txt
```

---

## 4. Features Used by the Model

The model uses approximately 40 features per stat (PTS, REB, AST share base features plus stat-specific additions).

**Rolling averages (majority of feature weight)**
Player performance averages over the last 5, 10, and 20 games for primary stats (PTS, REB, AST), plus shooting efficiency (FG%, 3P%, FT%), shot volume (FGA, FG3A, FTA), turnovers, and plus/minus.

**Rest and fatigue**
- `DAYS_REST` — days since last game
- `IS_BACK_TO_BACK` — flag for 0 rest days
- `IS_WELL_RESTED` — flag for 3+ days rest

**Home/away**
`IS_HOME` flag. Home players generally score slightly higher; the model learns this from historical data.

**Opponent defensive strength**
- `OPP_PTS_allowed_avg_L10` — points per game the opponent allows over their last 10 games
- `OPP_PTS_vs_POSITION_L10` — how the opponent defends against the player's position specifically

**Pace features**
- `TEAM_PACE_L10`, `OPP_PACE_L10` — recent pace for both teams
- `COMBINED_PACE` — sum of both team paces, a proxy for total possessions and scoring opportunities

**Return from injury**
- `IS_RETURNING` — flag for a player coming back after a gap
- `GAMES_SINCE_RETURN` — games played since returning
- `RETURN_GAME_NUMBER` — sequence number within the return (first game back vs third game back)

These features help the model discount players who may be on a minutes restriction or still shaking off rust.

**Teammate and opponent absences**
- `TEAMMATE_USAGE_ABSORBED` — estimate of how much scoring usage a player picks up when key teammates are out, based on proportional scoring share
- `TEAMMATES_OUT_COUNT` — number of key teammates (top-20% scorers on the team) absent
- `KEY_TEAMMATE_OUT` — binary flag when a star teammate (20+ PPG average) is missing
- `KEY_OPP_OUT` — binary flag when a key opposing player is out

**Foul features**
- `PFD_per36_avg_L10` — fouls drawn per 36 minutes; signals how often a player gets to the free throw line
- `HIGH_FT_DRAW` — flag for above-average foul drawing rate
- `PF_per36_avg_L10` — personal fouls per 36 minutes; foul trouble limits minutes and rebounding opportunities
- `FOUL_TROUBLE_RISK` — flag for elevated personal foul rate

---

## 5. Playoff Intelligence

Three systems activate automatically when playoff games are detected (game IDs starting with `004`):

**Playoff calibration**
The model is trained on regular season data and over-predicts in the playoffs due to slower pace and tighter defense. A larger offset is applied on top of the regular season calibration:

| | Regular Season | Playoffs |
|---|---|---|
| PTS | -0.50 | -2.00 total |
| REB | -0.10 | -0.10 total |
| AST | -0.05 | -0.25 total |

The additional playoff offset (`PTS -1.50`, `AST -0.20`) is derived from evaluating multiple playoff game days and measuring persistent bias.

**Playoff rotation filter**
Deep bench players and spot-minute contributors are automatically removed from output when their Vegas line is below 8.0 PTS. This keeps the prediction table focused on players with meaningful floor time. Players with no Vegas line are kept but flagged `ROTATION_UNCERTAIN=1`.

**Playoff elevation**
Some players reliably outperform their regular season averages in the playoffs (Jalen Brunson: +6.2 PTS historically; De'Aaron Fox: +9.7 PTS). Others regress, particularly young bench players stepping back into smaller roles.

Historical playoff game logs from the last three seasons (2022-23, 2023-24, 2024-25) are fetched at prediction time and compared against each player's regular season L20 rolling average to compute per-player elevation deltas. A 30% weight is applied:

```
PTS_PRED += PTS_ELEVATION * 0.30
```

This runs entirely at prediction time — no changes to training data.

---

## 6. Daily Usage

**Environment setup (first time only):**
Create a `.env` file in the project root:
```
ODDS_API_KEY=your_key_here
```
The Odds API key is required for Vegas line comparisons. A free tier key can be obtained at https://the-odds-api.com

**Start infrastructure:**
```bash
docker compose up -d
```

**Run predictions manually: (for 2025-26 season)**
```bash
venv\Scripts\python src\models\predict.py YYYY-MM-DD 2025-26
``` 

**Start dashboard: (run on flask)**
```bash
venv\Scripts\python src\dashboard\run.py
```
Then open `http://localhost:5000`

**Evaluate after games complete:**
```bash
venv\Scripts\python src\models\evaluate.py YYYY-MM-DD
```

---

## 7. Automated Scheduling

Three Airflow DAGs run sequentially each day, chained via file-based sensors:

| Time (UTC) | DAG | What it does |
|---|---|---|
| 9:00 AM | `ingest_dag` | Fetches player and team game logs for the current season, saves to `data/raw/` |
| 10:00 AM | `features_dag` | Waits for today's ingest file, then rebuilds the full feature matrix |
| 11:00 AM | `predict_dag` | Waits for today's feature file, then runs the full prediction pipeline |

Each DAG uses a `PythonSensor` that checks whether the upstream output file was written today (`date.fromtimestamp(os.path.getmtime(path)) == date.today()`), polling every 30 seconds with a 1-hour timeout. This approach is robust to schedule drift and does not depend on Airflow execution timestamp matching.

---

## 8. Model Architecture

**Why Ridge Regression over XGBoost for production predictions**

XGBoost was the original production model. After switching to Ridge Regression (with StandardScaler), evaluation results improved slightly and predictions became more stable — particularly for players with sparse or inconsistent recent game logs. XGBoost tended to overfit to recent hot/cold streaks. Ridge penalizes large coefficients and produces smoother, more conservative predictions that better reflect the underlying signal.

**XGBoost quantile models for confidence intervals**

Two XGBoost models are trained per stat at the p10 and p90 quantiles using `objective="reg:quantileerror"`. These produce a low/high range displayed as `PTS (p10-p90)` in the prediction table.

**Predictions clipped to p10-p90 bounds**

After calibration offsets are applied, each point prediction is clipped to stay within its own confidence interval. This prevents cases where the Ridge prediction lands outside the quantile bounds after calibration shifts.

**Calibration approach**

Calibration offsets are derived empirically by running `evaluate.py` on multiple game days and measuring the average prediction bias (mean of `PRED - ACTUAL`). Offsets are updated manually when enough evaluation days have accumulated to distinguish true bias from variance. Regular season and playoff calibration are tracked separately.

---

## 9. Evaluation System

`src/models/evaluate.py` grades predictions against actual box scores after games complete.

**Season type fallback**
The evaluator tries to fetch actuals in order: Regular Season -> PlayIn -> Playoffs. This ensures correct data is retrieved throughout the full calendar year without manual configuration.

**Outlier exclusion**
Players with an absolute PTS error greater than 15 are excluded from the clean metrics section. These are typically injury DNPs that were not caught by the injury filter, or extreme outlier performances. Outliers are listed separately for review.

**Metrics reported**
- Full set: PTS MAE and bias across all matched players
- Clean set: PTS/REB/AST MAE and bias with outliers removed
- Model vs Vegas: side-by-side PTS MAE comparison on players where both a Vegas line and actual result exist
- Top 10 best and worst predictions for the day
- Vegas disagreements: breakdown of flagged players (model vs Vegas diff > 3.0 PTS) and which side was closer

---

## 10. Roadmap

**Completed**
- Multi-season historical data (4 seasons, 105,000+ rows)
- Ridge Regression as production model
- Confidence intervals via XGBoost quantile models
- Vegas line comparison (Odds API integration)
- Real-time injury feed (ESPN API)
- Return from injury feature
- Teammate usage absorbed feature
- Positional matchup difficulty (OPP_PTS_vs_POSITION)
- Pace adjustment features
- Foul features (drawn and committed)
- Post-game evaluation system
- Prediction dashboard (Flask, two tabs: predictions and evaluation)
- Automated scheduling (Airflow DAGs with file-based sensors)
- Playoff rotation filter
- Playoff elevation feature (historical playoff performance)
- Playoff-specific calibration
- Dynamic calibration tuning (bias measured across multiple evaluation days)

**In Progress / Next Up**
- Accumulate first-round playoff data for 2025-26
- Refine playoff calibration offsets with more data
- Dynamic calibration (rolling average of recent bias, auto-adjusting)
- Travel fatigue feature (back-to-backs across time zones)
- Clutch performance history (late-game scoring tendencies)

**Future Ideas**
- Real-time lineup data integration (starting lineup confirmation before tip-off)
- Series-specific opponent adjustment (model learns how teams match up over a playoff series)
- Separate playoff model trained exclusively on playoff data
- Player embeddings: learn a dense vector representation for each player that captures latent characteristics (consistency, playing style, clutch tendency) beyond what rolling averages can express. Would improve variance estimation and help with small sample players. Likely implemented as a hybrid -- train an autoencoder on historical game logs to produce a 16-dimensional player vector, then feed those as additional features into the existing Ridge model.
- Bayesian Ridge Regression: a natural upgrade from plain Ridge that produces a full probability distribution over predictions rather than a point estimate. Confidence intervals would come directly from the model rather than needing separate XGBoost quantile models, simplifying the architecture while potentially improving calibration.
