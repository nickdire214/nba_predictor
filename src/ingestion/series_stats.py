import os
import unicodedata

import pandas as pd
from loguru import logger
from nba_api.stats.endpoints import scoreboardv3, playergamelogs


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").strip()


def get_active_series(game_date: str) -> list:
    """
    Returns list of (away_tricode, home_tricode) tuples for playoff games on game_date.
    Filters to gameIds starting with '004' (playoff).
    """
    try:
        board = scoreboardv3.ScoreboardV3(game_date=game_date)
        for df in board.get_data_frames():
            if "gameCode" in df.columns and "gameId" in df.columns:
                df = df[df["gameId"].astype(str).str.startswith("004")].copy()
                if df.empty:
                    return []
                df["away_tri"] = df["gameCode"].str.split("/").str[1].str[:3]
                df["home_tri"] = df["gameCode"].str.split("/").str[1].str[3:]
                pairs = list(zip(df["away_tri"], df["home_tri"]))
                logger.info(f"Active playoff series on {game_date}: {pairs}")
                return pairs
        return []
    except Exception as e:
        logger.warning(f"Could not fetch active series for {game_date}: {e}")
        return []


def get_series_adjustments(game_date: str, features_path: str = None) -> pd.DataFrame:
    """
    For each active playoff series on game_date, compute per-player in-series stats
    and delta vs their L10 baseline from the feature matrix.

    Returns DataFrame with columns:
        PLAYER_NAME_ASCII, SERIES_GAMES,
        SERIES_PTS_AVG, SERIES_REB_AVG, SERIES_AST_AVG,
        SERIES_PTS_DELTA, SERIES_REB_DELTA, SERIES_AST_DELTA

    SERIES_PTS_DELTA is non-zero only for players with SERIES_GAMES >= 2.
    """
    if features_path is None:
        features_path = os.path.join("data", "features", "player_features.csv")

    matchups = get_active_series(game_date)
    if not matchups:
        logger.warning(f"No active playoff series found for {game_date}.")
        return pd.DataFrame()

    # Fetch current-season playoff logs
    try:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable="2025-26",
            season_type_nullable="Playoffs",
        )
        playoff_df = logs.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed to fetch current-season playoff logs: {e}")
        return pd.DataFrame()

    if playoff_df.empty:
        logger.warning("No current-season playoff game logs returned.")
        return pd.DataFrame()

    playoff_df["PLAYER_NAME_ASCII"] = playoff_df["PLAYER_NAME"].apply(_ascii)

    # Filter to games belonging to today's active series matchups.
    # MATCHUP column has values like "BOS vs. PHI" or "PHI @ BOS".
    series_frames = []
    for away, home in matchups:
        teams = {away, home}
        mask = playoff_df["MATCHUP"].apply(lambda m: all(t in str(m) for t in teams))
        chunk = playoff_df[mask]
        if not chunk.empty:
            series_frames.append(chunk)
            logger.info(f"  {away} vs {home}: {len(chunk)} player-game rows")

    if not series_frames:
        logger.warning("No series game logs matched today's playoff matchups.")
        return pd.DataFrame()

    series_df = pd.concat(series_frames, ignore_index=True).drop_duplicates()
    logger.info(
        f"Total series logs: {len(series_df)} rows | "
        f"{series_df['PLAYER_NAME'].nunique()} players"
    )

    # Aggregate per player
    series_agg = (
        series_df.groupby("PLAYER_NAME_ASCII")
        .agg(
            SERIES_GAMES=("PTS", "count"),
            SERIES_PTS_AVG=("PTS", "mean"),
            SERIES_REB_AVG=("REB", "mean"),
            SERIES_AST_AVG=("AST", "mean"),
        )
        .round(2)
    )

    # Load L10 baseline from feature matrix
    if not os.path.exists(features_path):
        logger.warning(f"Feature matrix not found at {features_path} -- deltas will be 0.")
        series_agg["SERIES_PTS_DELTA"] = 0.0
        series_agg["SERIES_REB_DELTA"] = 0.0
        series_agg["SERIES_AST_DELTA"] = 0.0
        return series_agg.reset_index()

    feat_df = pd.read_csv(features_path).copy()
    feat_df["PLAYER_NAME_ASCII"] = feat_df["PLAYER_NAME"].apply(_ascii)

    baseline = (
        feat_df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME_ASCII")[["PTS_avg_L10", "REB_avg_L10", "AST_avg_L10"]]
        .last()
    )

    merged = series_agg.join(baseline, how="left")

    merged["SERIES_PTS_DELTA"] = (merged["SERIES_PTS_AVG"] - merged["PTS_avg_L10"]).fillna(0.0).round(2)
    merged["SERIES_REB_DELTA"] = (merged["SERIES_REB_AVG"] - merged["REB_avg_L10"]).fillna(0.0).round(2)
    merged["SERIES_AST_DELTA"] = (merged["SERIES_AST_AVG"] - merged["AST_avg_L10"]).fillna(0.0).round(2)

    # Zero out deltas for players with fewer than 2 series games
    few_games = merged["SERIES_GAMES"] < 2
    merged.loc[few_games, ["SERIES_PTS_DELTA", "SERIES_REB_DELTA", "SERIES_AST_DELTA"]] = 0.0

    result = merged[[
        "SERIES_GAMES",
        "SERIES_PTS_AVG", "SERIES_REB_AVG", "SERIES_AST_AVG",
        "SERIES_PTS_DELTA", "SERIES_REB_DELTA", "SERIES_AST_DELTA",
    ]].reset_index()

    n_qualified = int((result["SERIES_GAMES"] >= 2).sum())
    logger.info(
        f"Series adjustments ready: {len(result)} players total | "
        f"{n_qualified} with 2+ series games"
    )
    return result
