import pandas as pd
import numpy as np
import os
import json
import unicodedata
from datetime import datetime, timedelta
from loguru import logger
from nba_api.stats.endpoints import playergamelogs, teamgamelogs, scoreboardv3
import joblib
import xgboost as xgb
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingestion.roster import get_available_players
from src.ingestion.injuries import get_unavailable_players
from src.ingestion.odds import get_vegas_lines
from src.ingestion.playoff_stats import get_playoff_elevation
from src.features.engineer import (
    prepare_player_df,
    add_rolling_averages,
    add_rest_days,
    add_home_away,
    add_defensive_strength,
    add_absence_features,
    add_foul_features,
    add_player_embeddings,
    load_latest_raw,
)


# ─────────────────────────────────────────────
# Load trained models
# ─────────────────────────────────────────────

def load_models(is_playoff: bool = False) -> dict:
    """
    Load production models.
    When is_playoff=True, tries playoff Ridge models first before falling back.
    Priority overall: Playoff Ridge > Bayesian Ridge > Ridge > XGBoost.
    """
    playoff_files = {
        "PTS": os.path.join("models", "pts_model_playoff_ridge.pkl"),
        "REB": os.path.join("models", "reb_model_playoff_ridge.pkl"),
        "AST": os.path.join("models", "ast_model_playoff_ridge.pkl"),
    }
    bayesian_files = {
        "PTS": os.path.join("models", "pts_model_bayesian.pkl"),
        "REB": os.path.join("models", "reb_model_bayesian.pkl"),
        "AST": os.path.join("models", "ast_model_bayesian.pkl"),
    }
    ridge_files = {
        "PTS": os.path.join("models", "pts_model_ridge.pkl"),
        "REB": os.path.join("models", "reb_model_ridge.pkl"),
        "AST": os.path.join("models", "ast_model_ridge.pkl"),
    }
    xgb_files = {
        "PTS": os.path.join("models", "pts_model.json"),
        "REB": os.path.join("models", "reb_model.json"),
        "AST": os.path.join("models", "ast_model.json"),
    }

    use_playoff  = is_playoff and all(os.path.exists(p) for p in playoff_files.values())
    use_bayesian = all(os.path.exists(p) for p in bayesian_files.values())
    use_ridge    = all(os.path.exists(p) for p in ridge_files.values())

    if use_playoff:
        logger.info("Using Playoff Ridge models.")
        models = {}
        for stat, path in playoff_files.items():
            models[stat] = joblib.load(path)
            logger.info(f"Loaded {stat} Playoff Ridge model from {path}")
    elif use_bayesian:
        logger.info("Using Regular Season Bayesian Ridge models.")
        models = {}
        for stat, path in bayesian_files.items():
            models[stat] = joblib.load(path)
            logger.info(f"Loaded {stat} Bayesian Ridge model from {path}")
    elif use_ridge:
        logger.info("Using Regular Season Ridge models.")
        models = {}
        for stat, path in ridge_files.items():
            models[stat] = joblib.load(path)
            logger.info(f"Loaded {stat} Ridge model from {path}")
    else:
        logger.info("Using XGBoost production models.")
        models = {}
        for stat, path in xgb_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
            model = xgb.XGBRegressor()
            model.load_model(path)
            models[stat] = model
            logger.info(f"Loaded {stat} XGBoost model from {path}")

    return models, use_playoff


def load_quantile_models() -> dict:
    """
    Load lower (p10) and upper (p90) quantile models for each stat.
    Returns a dict keyed by e.g. 'PTS_lower', 'PTS_upper'.
    Gracefully skips any model file that doesn't exist yet.
    """
    quantile_files = {
        "PTS_lower": "pts_model_lower.json",
        "PTS_upper": "pts_model_upper.json",
        "REB_lower": "reb_model_lower.json",
        "REB_upper": "reb_model_upper.json",
        "AST_lower": "ast_model_lower.json",
        "AST_upper": "ast_model_upper.json",
    }
    q_models = {}
    for key, filename in quantile_files.items():
        path = os.path.join("models", filename)
        if not os.path.exists(path):
            logger.warning(f"Quantile model not found (skipping): {path}")
            continue
        model = xgb.XGBRegressor()
        model.load_model(path)
        q_models[key] = model
        logger.info(f"Loaded quantile model {key} from {path}")
    return q_models


# ─────────────────────────────────────────────
# Feature columns — must match train.py exactly
# ─────────────────────────────────────────────

BASE_FEATURES = [
    "DAYS_REST", "IS_BACK_TO_BACK", "IS_WELL_RESTED",
    "IS_HOME", "OPP_PTS_allowed_avg_L10", "OPP_PTS_vs_POSITION_L10",
    "TEAMMATE_USAGE_ABSORBED", "TEAMMATES_OUT_COUNT", "KEY_TEAMMATE_OUT", "KEY_OPP_OUT",
    "MIN_avg_L5", "MIN_avg_L10",
    "TEAM_PACE_L10", "OPP_PACE_L10", "COMBINED_PACE",
    "GAMES_SINCE_RETURN", "IS_RETURNING", "RETURN_GAME_NUMBER",
] + [f"EMB_{i}" for i in range(16)]

STAT_FEATURES = {
    "PTS": BASE_FEATURES + [
        "PTS_avg_L5", "PTS_avg_L10", "PTS_avg_L20",
        "FGA_avg_L5", "FGA_avg_L10",
        "FG_PCT_avg_L5", "FG_PCT_avg_L10",
        "FG3A_avg_L5", "FG3_PCT_avg_L5",
        "FTA_avg_L5", "FT_PCT_avg_L5",
        "PLUS_MINUS_avg_L5",
        "PFD_per36_avg_L10", "HIGH_FT_DRAW",
    ],
    "REB": BASE_FEATURES + [
        "REB_avg_L5", "REB_avg_L10", "REB_avg_L20",
        "OREB_avg_L5", "OREB_avg_L10",
        "DREB_avg_L5", "DREB_avg_L10",
        "FGA_avg_L5", "PLUS_MINUS_avg_L5",
        "PF_per36_avg_L10", "FOUL_TROUBLE_RISK",
    ],
    "AST": BASE_FEATURES + [
        "AST_avg_L5", "AST_avg_L10", "AST_avg_L20",
        "TOV_avg_L5", "TOV_avg_L10",
        "PTS_avg_L5", "FGA_avg_L5",
        "PLUS_MINUS_avg_L5",
        "PF_per36_avg_L10", "FOUL_TROUBLE_RISK",
    ],
}


# ─────────────────────────────────────────────
# Fetch today's games
# ─────────────────────────────────────────────

def get_todays_games(game_date: str = None) -> pd.DataFrame:
    """
    Fetch today's scheduled games from the NBA scoreboard.
    Returns a DataFrame with home and away team info.
    game_date format: 'YYYY-MM-DD' — defaults to today.
    """
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Fetching games for {game_date}...")

    try:
        board = scoreboardv3.ScoreboardV3(game_date=game_date)
        all_dfs = board.get_data_frames()

        # Log all available dataframes so we can find the right one
        for i, df in enumerate(all_dfs):
            logger.info(f"DataFrame[{i}] columns: {list(df.columns)[:6]}... rows: {len(df)}")

        # Find the dataframe that contains team ID columns
        games_df = None
        for df in all_dfs:
            cols = [c.lower() for c in df.columns]
            if any("teamid" in c or "team_id" in c for c in cols):
                games_df = df
                logger.info(f"Found games dataframe with columns: {list(df.columns)}")
                break

        if games_df is None or games_df.empty:
            logger.warning(f"No games found for {game_date}.")
            return pd.DataFrame()

        # Enrich with home/away info from gameCode (format: YYYYMMDD/AWAYHHOME)
        # e.g. "20260406/NYKATL" -> away=NYK, home=ATL
        for df in all_dfs:
            if "gameCode" in df.columns and "gameId" in df.columns:
                meta = df[["gameId", "gameCode"]].copy()
                meta["away_tri"] = meta["gameCode"].str.split("/").str[1].str[:3]
                meta["home_tri"] = meta["gameCode"].str.split("/").str[1].str[3:]
                games_df = games_df.merge(meta, on="gameId", how="left")
                logger.info("Enriched games_df with home/away tricode from gameCode.")
                break

    except Exception as e:
        logger.error(f"Failed to fetch scoreboard: {e}")
        raise

    logger.info(f"Found {len(games_df)} team rows on {game_date}.")
    return games_df


# ─────────────────────────────────────────────
# Build prediction input for today's players
# ─────────────────────────────────────────────

def build_prediction_input(game_date: str = None):
    """
    Load the full season feature matrix and filter down to players
    whose teams are playing today.
    Returns (player_df, playing_team_ids).
    """
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")

    features_path = os.path.join("data", "features", "player_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError("Feature matrix not found. Run engineer.py first.")

    df = pd.read_csv(features_path).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = add_player_embeddings(df)

    games_df = get_todays_games(game_date)
    if games_df.empty:
        return pd.DataFrame(), set()

    playing_team_ids = set()
    if "teamId" in games_df.columns:
        playing_team_ids.update(games_df["teamId"].tolist())
        logger.info(f"Parsed {len(playing_team_ids)} team IDs from teamId column.")
    else:
        logger.warning("Could not parse team IDs - using all teams.")
        logger.warning(f"Available columns: {list(games_df.columns)}")

    if playing_team_ids:
        df = df[df["TEAM_ID"].isin(playing_team_ids)]

    latest_per_player = (
        df.sort_values("GAME_DATE")
        .groupby("PLAYER_ID")
        .last()
        .reset_index()
        .copy()
    )

    latest_per_player = latest_per_player.drop(
        columns=["DAYS_REST", "IS_BACK_TO_BACK", "IS_WELL_RESTED"],
        errors="ignore"
    )

    today = pd.Timestamp(game_date)
    days_rest = ((today - latest_per_player["GAME_DATE"]).dt.days - 1).clip(lower=0, upper=14)
    latest_per_player["DAYS_REST"] = days_rest
    latest_per_player["IS_BACK_TO_BACK"] = (days_rest == 0).astype(int)
    latest_per_player["IS_WELL_RESTED"] = (days_rest >= 3).astype(int)

    # Overwrite MATCHUP and IS_HOME with tonight's actual game data
    if "away_tri" in games_df.columns and "home_tri" in games_df.columns:
        matchup_map = {}
        is_home_map = {}
        for _, row in games_df.iterrows():
            tid = row["teamId"]
            if row["teamTricode"] == row["home_tri"]:
                matchup_map[tid] = f"{row['teamTricode']} vs. {row['away_tri']}"
                is_home_map[tid] = 1
            else:
                matchup_map[tid] = f"{row['teamTricode']} @ {row['home_tri']}"
                is_home_map[tid] = 0
        latest_per_player["MATCHUP"] = (
            latest_per_player["TEAM_ID"].map(matchup_map)
            .fillna(latest_per_player["MATCHUP"])
        )
        latest_per_player["IS_HOME"] = (
            latest_per_player["TEAM_ID"].map(is_home_map)
            .fillna(latest_per_player["IS_HOME"])
        ).astype(int)
        logger.info("Overwrote MATCHUP and IS_HOME with tonight's actual game data.")

    latest_per_player = latest_per_player.copy()
    logger.info(f"Built prediction input for {len(latest_per_player)} players.")
    return latest_per_player, playing_team_ids


###################################################################################


def filter_available_players(
    player_df: pd.DataFrame,
    team_ids: list,
    season: str = "2025-26"
) -> pd.DataFrame:
    """
    Filter prediction input to only players who are
    available to play tonight based on roster and
    recent activity checks.
    """
    if not team_ids:
        logger.warning("No team IDs provided - skipping availability filter.")
        return player_df

    available_ids = get_available_players(
        team_ids=list(team_ids),
        season=season,
        n_games=3
    )

    if not available_ids:
        logger.warning("Could not determine availability - using full player pool.")
        return player_df

    before = len(player_df)
    filtered = player_df[player_df["PLAYER_ID"].isin(available_ids)].copy()
    removed = before - len(filtered)
    logger.info(f"Roster/activity filter removed {removed} players - "
                f"{len(filtered)} remaining.")

    # Injury filter — remove players ruled Out per ESPN injury report
    try:
        unavailable = get_unavailable_players()
        before_injury = len(filtered)
        filtered = filtered[~filtered["PLAYER_NAME"].isin(unavailable)].copy()
        injury_removed = before_injury - len(filtered)
        logger.info(f"Injury filter removed {injury_removed} players (Out) - "
                    f"{len(filtered)} remaining.")
    except Exception as e:
        logger.warning(f"Injury filter skipped (ESPN fetch failed): {e}")

    return filtered


# ─────────────────────────────────────────────
# Generate predictions
# ─────────────────────────────────────────────

def generate_predictions(models: dict, player_df: pd.DataFrame,
                         q_models: dict = None) -> pd.DataFrame:
    """
    Run each model and attach predictions to the player DataFrame.
    If q_models is provided, also generates p10/p90 interval bounds.
    Returns a clean output DataFrame with one row per player.
    """
    results = player_df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
                          "MATCHUP", "DAYS_REST", "IS_HOME"]].copy().reset_index(drop=True)

    if q_models is None:
        q_models = {}

    for stat, model in models.items():
        feature_cols = STAT_FEATURES[stat]
        available = [f for f in feature_cols if f in player_df.columns]
        missing = set(feature_cols) - set(available)
        if missing:
            logger.warning(f"{stat} missing features: {missing}")

        X = player_df[available].fillna(0).reset_index(drop=True)

        try:
            preds = model.predict(X).astype(float)
            results[f"{stat}_PRED"] = np.round(preds, 1)
        except Exception as e:
            logger.error(f"Prediction failed for {stat}: {e}")
            results[f"{stat}_PRED"] = np.nan

        for bound, suffix in [("lower", "LOW"), ("upper", "HIGH")]:
            key = f"{stat}_{bound}"
            col = f"{stat}_{suffix}"
            if key in q_models:
                try:
                    q_preds = q_models[key].predict(X).astype(float)
                    results[col] = np.round(q_preds, 1)
                except Exception as e:
                    logger.error(f"Quantile prediction failed for {key}: {e}")
                    results[col] = np.nan
            else:
                results[col] = np.nan

    return results


# ─────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────

# Regular season calibration offsets (derived from evaluation days).
_CALIBRATION = {
    "PTS": 0.50,
    "REB": 0.10,
    "AST": 0.05,
}

# Additional offset for playoff games using the regular season model.
# The regular season model over-predicts more in playoffs.
_PLAYOFF_EXTRA_RS = {
    "PTS": 1.50,
    "REB": 0.00,
    "AST": 0.20,
}

# Reduced additional offset for playoff games using the playoff model.
# The playoff model already accounts for playoff scoring patterns.
_PLAYOFF_EXTRA_PO = {
    "PTS": 0.50,
    "REB": 0.00,
    "AST": 0.10,
}


def apply_calibration(
    results: pd.DataFrame,
    is_playoff: bool = False,
    use_playoff_model: bool = False,
) -> pd.DataFrame:
    """
    Subtract known positive bias from predictions and quantile bounds,
    then clip each prediction to stay within its own p10-p90 interval.
    When is_playoff=True, applies an additional playoff offset.
    When use_playoff_model=True the reduced offset is used since the
    playoff model already captures playoff scoring patterns.
    """
    if is_playoff:
        extra = _PLAYOFF_EXTRA_PO if use_playoff_model else _PLAYOFF_EXTRA_RS
    else:
        extra = {"PTS": 0.0, "REB": 0.0, "AST": 0.0}

    offsets = {stat: _CALIBRATION[stat] + extra[stat] for stat in _CALIBRATION}

    if is_playoff:
        label = "playoff model" if use_playoff_model else "regular model"
        logger.info(
            f"Playoff calibration ({label}): PTS -{offsets['PTS']:.2f} "
            f"REB -{offsets['REB']:.2f} "
            f"AST -{offsets['AST']:.2f}"
        )
    else:
        logger.info(
            f"Regular season calibration: PTS -{offsets['PTS']:.2f} "
            f"REB -{offsets['REB']:.2f} "
            f"AST -{offsets['AST']:.2f}"
        )

    results = results.copy()

    # Step 1: shift all three columns by calibration offset
    for stat, offset in offsets.items():
        for col in [f"{stat}_PRED", f"{stat}_LOW", f"{stat}_HIGH"]:
            if col in results.columns:
                results[col] = (results[col] - offset).clip(lower=0.0).round(1)

    # Step 2: clip predictions to [p10, p90] where bounds exist
    n_above = n_below = 0
    for stat in _CALIBRATION:
        pred_col = f"{stat}_PRED"
        lo_col   = f"{stat}_LOW"
        hi_col   = f"{stat}_HIGH"
        if pred_col not in results.columns:
            continue
        has_bounds = results[lo_col].notna() & results[hi_col].notna()
        if not has_bounds.any():
            continue

        above = has_bounds & (results[pred_col] > results[hi_col])
        below = has_bounds & (results[pred_col] < results[lo_col])
        n_above += int(above.sum())
        n_below += int(below.sum())

        results.loc[above, pred_col] = results.loc[above, hi_col]
        results.loc[below, pred_col] = results.loc[below, lo_col]

    logger.info(f"Clipped {n_above} predictions above p90, {n_below} below p10.")
    return results


# ─────────────────────────────────────────────
# Save and display predictions
# ─────────────────────────────────────────────

def merge_vegas_lines(results: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch Vegas player prop lines and left-join onto the predictions DataFrame.
    Adds VEGAS_PTS, VEGAS_REB, VEGAS_AST and the three DIFF columns.
    Players without Vegas lines keep NaN in those columns.
    """
    vegas = get_vegas_lines()
    if vegas.empty:
        logger.warning("No Vegas lines available -- skipping merge.")
        for col in ["VEGAS_PTS", "VEGAS_REB", "VEGAS_AST",
                    "PTS_DIFF", "REB_DIFF", "AST_DIFF"]:
            results[col] = np.nan
        return results

    results = results.merge(vegas, on="PLAYER_NAME", how="left")

    results["PTS_DIFF"] = (results["PTS_PRED"] - results["VEGAS_PTS"]).round(1)
    results["REB_DIFF"] = (results["REB_PRED"] - results["VEGAS_REB"]).round(1)
    results["AST_DIFF"] = (results["AST_PRED"] - results["VEGAS_AST"]).round(1)

    matched = results["VEGAS_PTS"].notna().sum()
    logger.info(f"Vegas lines merged: {matched}/{len(results)} players matched.")
    return results


def _playoff_elevation_weight(games: int) -> float:
    if games >= 30:
        return 0.50
    if games >= 15:
        return 0.40
    if games >= 5:
        return 0.30
    if games > 0:
        return 0.15
    return 0.00


def apply_playoff_elevation(results: pd.DataFrame) -> pd.DataFrame:
    """
    Boost predictions for players with a positive playoff elevation signal.
    Applies a tiered weight based on career playoff games played.
    Also adjusts LOW/HIGH bounds by the same amount.
    Only called when is_playoff_date() is True.
    """
    elev = get_playoff_elevation()
    if elev.empty:
        logger.warning("Playoff elevation table is empty -- skipping elevation adjustment.")
        return results

    results = results.copy()

    import unicodedata as _ud
    def _ascii(n):
        return _ud.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip()

    results["_name_ascii"] = results["PLAYER_NAME"].apply(_ascii)

    tier_pts = {">= 30": [], "15-29": [], "5-14": [], "< 5": []}

    for idx, row in results.iterrows():
        key = row["_name_ascii"]
        if key not in elev.index:
            continue

        player_elev = elev.loc[key]
        games = int(player_elev["PLAYOFF_GAMES"])
        weight = _playoff_elevation_weight(games)
        if weight == 0.0:
            continue

        if games >= 30:
            bucket = ">= 30"
        elif games >= 15:
            bucket = "15-29"
        elif games >= 5:
            bucket = "5-14"
        else:
            bucket = "< 5"

        for stat, elev_col in [("PTS", "PTS_ELEVATION"), ("REB", "REB_ELEVATION"), ("AST", "AST_ELEVATION")]:
            delta = float(player_elev[elev_col]) * weight
            if stat == "PTS":
                tier_pts[bucket].append(delta)
            for col in [f"{stat}_PRED", f"{stat}_LOW", f"{stat}_HIGH"]:
                if col in results.columns and pd.notna(results.at[idx, col]):
                    results.at[idx, col] = round(max(0.0, float(results.at[idx, col]) + delta), 1)

    results = results.drop(columns=["_name_ascii"])

    total = sum(len(v) for v in tier_pts.values())
    logger.info(f"Playoff elevation applied: {total} players")

    labels = [
        (">= 30", "30+ games"),
        ("15-29", "15-29 games"),
        ("5-14",  "5-14 games"),
    ]
    for key, label in labels:
        vals = tier_pts[key]
        if vals:
            avg = sum(vals) / len(vals)
            logger.info(f"  {label}: {len(vals)} players avg {avg:+.1f} pts")

    return results


def _series_weight(games: int) -> float:
    if games >= 5:
        return 0.25
    if games == 4:
        return 0.20
    if games == 3:
        return 0.15
    if games == 2:
        return 0.10
    return 0.00


def apply_series_adjustment(results: pd.DataFrame, game_date: str) -> pd.DataFrame:
    """
    For each player with 2+ games in the current series, apply a tiered weight
    (based on series games played) to their in-series PTS/REB/AST delta.
    Only called when is_playoff_date() is True.
    """
    from src.ingestion.series_stats import get_series_adjustments

    series = get_series_adjustments(game_date)
    if series.empty:
        logger.warning("Series adjustments empty -- skipping series adjustment.")
        return results

    results = results.copy()

    import unicodedata as _ud
    def _ascii(n):
        return _ud.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip()

    results["_name_ascii"] = results["PLAYER_NAME"].apply(_ascii)
    series_idx = series.set_index("PLAYER_NAME_ASCII")

    tier_pts = {"5+": [], "4": [], "3": [], "2": []}

    for idx, row in results.iterrows():
        key = row["_name_ascii"]
        if key not in series_idx.index:
            continue

        player_series = series_idx.loc[key]
        games = int(player_series["SERIES_GAMES"])
        weight = _series_weight(games)
        if weight == 0.0:
            continue

        if games >= 5:
            bucket = "5+"
        elif games == 4:
            bucket = "4"
        elif games == 3:
            bucket = "3"
        else:
            bucket = "2"

        for stat, delta_col in [
            ("PTS", "SERIES_PTS_DELTA"),
            ("REB", "SERIES_REB_DELTA"),
            ("AST", "SERIES_AST_DELTA"),
        ]:
            delta = float(player_series[delta_col]) * weight
            if stat == "PTS":
                tier_pts[bucket].append(delta)
            for col in [f"{stat}_PRED", f"{stat}_LOW", f"{stat}_HIGH"]:
                if col in results.columns and pd.notna(results.at[idx, col]):
                    results.at[idx, col] = round(max(0.0, float(results.at[idx, col]) + delta), 1)

    results = results.drop(columns=["_name_ascii"])

    total = sum(len(v) for v in tier_pts.values())
    logger.info(f"Series adjustment applied: {total} players")

    labels = [("5+", "5+ games"), ("4", "4 games"), ("3", "3 games"), ("2", "2 games")]
    for key, label in labels:
        vals = tier_pts[key]
        if vals:
            avg = sum(vals) / len(vals)
            logger.info(f"  {label}: {len(vals)} players avg {avg:+.1f} pts")

    return results


def is_playoff_date(game_date: str = None) -> bool:
    """
    Return True if the games on game_date are playoff games.
    Playoff game IDs start with '004'; regular season IDs start with '002'.
    """
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")
    try:
        games_df = get_todays_games(game_date)
        if games_df.empty or "gameId" not in games_df.columns:
            return False
        return bool(games_df["gameId"].astype(str).str.startswith("004").any())
    except Exception as e:
        logger.warning(f"Could not determine playoff status: {e}")
        return False


def apply_playoff_rotation_filter(results: pd.DataFrame) -> pd.DataFrame:
    """
    Playoff-only filter: remove players where Vegas has set a line below 8.0 PTS
    (deep bench / DNP risk). Players with no Vegas line are kept but flagged
    ROTATION_UNCERTAIN=1 so the dashboard can highlight them.
    """
    results = results.copy()
    results["ROTATION_UNCERTAIN"] = 0

    has_vegas = results["VEGAS_PTS"].notna()
    low_vegas = has_vegas & (results["VEGAS_PTS"] < 8.0)
    no_vegas  = ~has_vegas

    n_removed = int(low_vegas.sum())
    results.loc[no_vegas, "ROTATION_UNCERTAIN"] = 1
    n_uncertain = int(no_vegas.sum())

    logger.info(
        f"Playoff rotation filter removed {n_removed} players with Vegas lines under 8.0"
    )
    logger.info(
        f"Playoff rotation filter flagged {n_uncertain} players as ROTATION_UNCERTAIN (no Vegas line)"
    )

    return results[~low_vegas].copy()


def save_predictions(results: pd.DataFrame, game_date: str, suffix: str = ""):
    """Save prediction output to data/predictions/"""
    out_dir = os.path.join("data", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"predictions_{game_date}{suffix}.csv"
    filepath = os.path.join(out_dir, filename)
    results.to_csv(filepath, index=False)
    logger.info(f"Saved predictions to {filepath}")
    return filepath


def _fmt_diff(val) -> str:
    """Format a diff value as '+X.X' / '-X.X' / '   -' if NaN."""
    if pd.isna(val):
        return "   -"
    return f"{val:+.1f}"


def _fmt_line(val) -> str:
    """Format a Vegas line, or '-' if NaN."""
    if pd.isna(val):
        return "  -"
    return f"{val:.1f}"


def display_predictions(results: pd.DataFrame, min_pts: float = 8.0, game_date: str = None):
    """
    Print model vs Vegas comparison table sorted by abs(PTS_DIFF) descending.
    Only shows players where at least one Vegas line exists.
    Falls back to a plain predictions table if no Vegas lines were merged.
    Also prints a high-conviction disagreements section.
    """
    display_date = game_date or datetime.now().strftime("%Y-%m-%d")
    try:
        formatted_date = datetime.strptime(display_date, "%Y-%m-%d").strftime("%B %d, %Y")
    except Exception:
        formatted_date = display_date

    has_vegas = "VEGAS_PTS" in results.columns and results["VEGAS_PTS"].notna().any()

    has_intervals = ("PTS_LOW" in results.columns and results["PTS_LOW"].notna().any())

    def _rng(row, stat):
        lo = row.get(f"{stat}_LOW")
        hi = row.get(f"{stat}_HIGH")
        if pd.isna(lo) or pd.isna(hi):
            return ""
        return f"({int(round(lo))}-{int(round(hi))})"

    # ── plain table (no Vegas data) ──────────────────────────────────────
    if not has_vegas:
        plain = results[results["PTS_PRED"] >= min_pts].copy()
        plain = plain.sort_values("PTS_PRED", ascending=False)
        plain["REST"] = plain["DAYS_REST"].apply(lambda x: "B2B" if x == 0 else f"{int(x)}d")
        plain["LOC"]  = plain["IS_HOME"].apply(lambda x: "HOME" if x else "AWAY")

        if has_intervals:
            W = 90
            print(f"\n{'='*W}")
            print(f"  NBA Predictions - {formatted_date}")
            print(f"{'='*W}")
            print(f"{'Player':<24} {'Team':<5} {'Loc':<5} {'Rest':<5}"
                  f"  {'PTS (range)':<15} {'REB (range)':<15} {'AST (range)':<15}")
            print(f"{'-'*W}")
            for _, row in plain.iterrows():
                name = unicodedata.normalize("NFKD", str(row["PLAYER_NAME"])).encode("ascii", "ignore").decode("ascii")
                name = name[:23]
                pts_str = f"{row['PTS_PRED']:.1f} {_rng(row, 'PTS')}"
                reb_str = f"{row['REB_PRED']:.1f} {_rng(row, 'REB')}"
                ast_str = f"{row['AST_PRED']:.1f} {_rng(row, 'AST')}"
                rest = "B2B" if row["DAYS_REST"] == 0 else f"{int(row['DAYS_REST'])}d"
                loc  = "HOME" if row["IS_HOME"] else "AWAY"
                print(f"{name:<24} {str(row['TEAM_ABBREVIATION']):<5} {loc:<5} {rest:<5}"
                      f"  {pts_str:<15} {reb_str:<15} {ast_str:<15}")
            print(f"{'='*W}")
            print(f"  {len(plain)} players | min {min_pts} predicted PTS | range = p10-p90")
            print(f"{'='*W}\n")
        else:
            print(f"\n{'='*72}")
            print(f"  NBA Predictions - {formatted_date}")
            print(f"{'='*72}")
            print(f"{'Player':<25} {'Team':<5} {'Matchup':<14} {'Loc':<5} "
                  f"{'Rest':<5} {'PTS':>5} {'REB':>5} {'AST':>5}")
            print(f"{'-'*72}")
            for _, row in plain.iterrows():
                matchup = str(row["MATCHUP"])
                matchup = matchup[-13:] if len(matchup) > 13 else matchup
                name = unicodedata.normalize("NFKD", str(row["PLAYER_NAME"])).encode("ascii", "ignore").decode("ascii")
                print(f"{name:<25} {str(row['TEAM_ABBREVIATION']):<5} {matchup:<14} "
                      f"{row['LOC']:<5} {row['REST']:<5} "
                      f"{row['PTS_PRED']:>5} {row['REB_PRED']:>5} {row['AST_PRED']:>5}")
            print(f"{'='*72}")
            print(f"  {len(plain)} players | min {min_pts} predicted PTS")
            print(f"{'='*72}\n")
        return

    # ── Vegas comparison table ────────────────────────────────────────────
    # Only rows with at least one Vegas line
    df = results[results["VEGAS_PTS"].notna() | results["VEGAS_REB"].notna()].copy()
    df = df[df["PTS_PRED"] >= min_pts].copy()

    # Sort by abs PTS_DIFF descending; NaN diffs go to the bottom
    df["_abs_diff"] = df["PTS_DIFF"].abs().fillna(-1)
    df = df.sort_values("_abs_diff", ascending=False).drop(columns=["_abs_diff"])

    W = 100 if has_intervals else 88
    print(f"\n{'='*W}")
    print(f"  NBA Predictions vs Vegas - {formatted_date}")
    print(f"{'='*W}")
    if has_intervals:
        print(
            f"{'Player':<24} {'Team':<5}"
            f" {'PTS (range)':<16} {'PTS_Vegas':>9} {'PTS_Diff':>8}"
            f" {'REB (range)':<16} {'REB_Vegas':>9} {'REB_Diff':>8}"
            f"  {'Flg'}"
        )
    else:
        print(
            f"{'Player':<24} {'Team':<5}"
            f" {'PTS_Pred':>8} {'PTS_Vegas':>9} {'PTS_Diff':>8}"
            f" {'REB_Pred':>8} {'REB_Vegas':>9} {'REB_Diff':>8}"
            f"  {'Flag'}"
        )
    print(f"{'-'*W}")

    for _, row in df.iterrows():
        name = unicodedata.normalize("NFKD", str(row["PLAYER_NAME"])).encode("ascii", "ignore").decode("ascii")
        name = name[:23]
        flag = "*" if (not pd.isna(row.get("PTS_DIFF")) and abs(row["PTS_DIFF"]) > 3.0) else " "
        if has_intervals:
            pts_str = f"{row['PTS_PRED']:.1f} {_rng(row, 'PTS')}"
            reb_str = f"{row['REB_PRED']:.1f} {_rng(row, 'REB')}"
            print(
                f"{name:<24} {str(row['TEAM_ABBREVIATION']):<5}"
                f" {pts_str:<16} {_fmt_line(row['VEGAS_PTS']):>9} {_fmt_diff(row['PTS_DIFF']):>8}"
                f" {reb_str:<16} {_fmt_line(row['VEGAS_REB']):>9} {_fmt_diff(row['REB_DIFF']):>8}"
                f"  {flag}"
            )
        else:
            print(
                f"{name:<24} {str(row['TEAM_ABBREVIATION']):<5}"
                f" {row['PTS_PRED']:>8.1f} {_fmt_line(row['VEGAS_PTS']):>9} {_fmt_diff(row['PTS_DIFF']):>8}"
                f" {row['REB_PRED']:>8.1f} {_fmt_line(row['VEGAS_REB']):>9} {_fmt_diff(row['REB_DIFF']):>8}"
                f"  {flag}"
            )

    print(f"{'='*W}")
    print(f"  {len(df)} players with Vegas lines | min {min_pts} predicted PTS | * = |PTS diff| > 3.0")
    if has_intervals:
        print(f"  range = p10-p90 interval")
    print(f"{'='*W}\n")

    # ── High-conviction disagreements ─────────────────────────────────────
    big = df[df["PTS_DIFF"].abs() > 3.0].copy()
    if big.empty:
        return

    higher = big[big["PTS_DIFF"] > 0].sort_values("PTS_DIFF", ascending=False)
    lower  = big[big["PTS_DIFF"] < 0].sort_values("PTS_DIFF")

    print(f"{'='*W}")
    print(f"  High-conviction disagreements (|PTS diff| > 3.0)")
    print(f"{'='*W}")

    if not higher.empty:
        print("  Model HIGHER than Vegas (potential undervalued):")
        for _, row in higher.iterrows():
            name = unicodedata.normalize("NFKD", str(row["PLAYER_NAME"])).encode("ascii", "ignore").decode("ascii")
            print(f"    {name}: model {row['PTS_PRED']:.1f} vs Vegas {row['VEGAS_PTS']:.1f} ({row['PTS_DIFF']:+.1f})")

    if not lower.empty:
        print("  Model LOWER than Vegas (potential overvalued):")
        for _, row in lower.iterrows():
            name = unicodedata.normalize("NFKD", str(row["PLAYER_NAME"])).encode("ascii", "ignore").decode("ascii")
            print(f"    {name}: model {row['PTS_PRED']:.1f} vs Vegas {row['VEGAS_PTS']:.1f} ({row['PTS_DIFF']:+.1f})")

    print(f"{'='*W}\n")




# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    game_date        = sys.argv[1] if len(sys.argv) > 1 else None
    season           = sys.argv[2] if len(sys.argv) > 2 else "2025-26"
    no_playoff_model = "--no-playoff-model" in sys.argv

    date_str = game_date or datetime.now().strftime("%Y-%m-%d")
    playoff  = is_playoff_date(date_str)

    models, used_playoff_model = load_models(is_playoff=(playoff and not no_playoff_model))
    q_models = load_quantile_models()
    player_df, playing_team_ids = build_prediction_input(game_date=game_date)

    if player_df.empty:
        logger.warning("No players to predict for. Check if there are games today.")
    else:
        player_df = filter_available_players(
            player_df,
            team_ids=playing_team_ids,
            season=season
        )

        if player_df.empty:
            logger.warning("No available players after filtering.")
        else:
            results = generate_predictions(models, player_df, q_models=q_models)
            results = apply_calibration(
                results,
                is_playoff=playoff,
                use_playoff_model=used_playoff_model,
            )
            results = merge_vegas_lines(results)
            if playoff:
                logger.info("Playoff games detected -- applying elevation, series adjustment, and rotation filter.")
                results = apply_playoff_elevation(results)
                results = apply_series_adjustment(results, date_str)
                results = apply_playoff_rotation_filter(results)
            else:
                logger.info("Regular season games -- skipping playoff adjustments.")
            suffix = "_regular_model" if no_playoff_model else ""
            save_predictions(results, date_str, suffix=suffix)
            display_predictions(results, min_pts=8.0, game_date=date_str)
