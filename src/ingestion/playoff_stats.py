import os
import unicodedata

import pandas as pd
from loguru import logger

from src.ingestion.nba_stats import fetch_playoff_gamelogs

_PLAYOFF_SEASONS = ["2022-23", "2023-24", "2024-25"]


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").strip()


def get_playoff_elevation(
    seasons: list = None,
    features_path: str = None,
) -> pd.DataFrame:
    """
    Build a per-player playoff elevation table.

    For each player computes:
        PLAYOFF_PTS_AVG   mean PTS across all playoff games in seasons
        PLAYOFF_REB_AVG   mean REB across all playoff games in seasons
        PLAYOFF_AST_AVG   mean AST across all playoff games in seasons
        PLAYOFF_GAMES     total playoff games played
        PTS_ELEVATION     PLAYOFF_PTS_AVG minus regular-season L20 average
        REB_ELEVATION     same for rebounds
        AST_ELEVATION     same for assists

    Players with no playoff history get 0 for all columns.
    Returns a DataFrame indexed by ASCII player name.
    """
    if seasons is None:
        seasons = _PLAYOFF_SEASONS
    if features_path is None:
        features_path = os.path.join("data", "features", "player_features.csv")

    # ── Playoff game logs ─────────────────────────────────────────────────────
    logger.info("Fetching playoff game logs for elevation calculation...")
    playoff_df = fetch_playoff_gamelogs(seasons=seasons)

    if playoff_df.empty:
        logger.warning("No playoff data returned -- elevation table will be all zeros.")
        return pd.DataFrame(columns=[
            "PLAYOFF_GAMES", "PLAYOFF_PTS_AVG", "PLAYOFF_REB_AVG", "PLAYOFF_AST_AVG",
            "PTS_ELEVATION", "REB_ELEVATION", "AST_ELEVATION",
        ])

    playoff_df["PLAYER_NAME_ASCII"] = playoff_df["PLAYER_NAME"].apply(_ascii)

    playoff_agg = (
        playoff_df.groupby("PLAYER_NAME_ASCII")
        .agg(
            PLAYOFF_GAMES=("PTS", "count"),
            PLAYOFF_PTS_AVG=("PTS", "mean"),
            PLAYOFF_REB_AVG=("REB", "mean"),
            PLAYOFF_AST_AVG=("AST", "mean"),
        )
        .round(2)
    )

    logger.info(f"Playoff history computed for {len(playoff_agg)} players.")

    # ── Regular-season baseline from feature matrix ───────────────────────────
    if not os.path.exists(features_path):
        logger.warning(f"Feature matrix not found at {features_path} -- elevation vs baseline unavailable.")
        playoff_agg["PTS_ELEVATION"] = 0.0
        playoff_agg["REB_ELEVATION"] = 0.0
        playoff_agg["AST_ELEVATION"] = 0.0
        return playoff_agg

    feat_df = pd.read_csv(features_path)
    feat_df["PLAYER_NAME_ASCII"] = feat_df["PLAYER_NAME"].apply(_ascii)

    # Use the last row per player (most recent game) and read their L20 rolling averages
    # as the regular-season baseline
    baseline_cols = ["PLAYER_NAME_ASCII", "PTS_avg_L20", "REB_avg_L20", "AST_avg_L20"]
    available = [c for c in baseline_cols if c in feat_df.columns]
    if len(available) < 4:
        logger.warning("L20 average columns missing from feature matrix -- using L10 or L5 fallback.")
        for stat in ("PTS", "REB", "AST"):
            for win in ("L10", "L5"):
                col = f"{stat}_avg_{win}"
                if col in feat_df.columns:
                    feat_df[f"{stat}_avg_L20"] = feat_df[col]
                    break

    baseline = (
        feat_df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME_ASCII")[["PTS_avg_L20", "REB_avg_L20", "AST_avg_L20"]]
        .last()
        .rename(columns={
            "PTS_avg_L20": "RS_PTS_AVG",
            "REB_avg_L20": "RS_REB_AVG",
            "AST_avg_L20": "RS_AST_AVG",
        })
    )

    logger.info(f"Regular-season baseline computed for {len(baseline)} players.")

    # ── Merge and compute elevation ───────────────────────────────────────────
    merged = playoff_agg.join(baseline, how="left")

    merged["PTS_ELEVATION"] = (merged["PLAYOFF_PTS_AVG"] - merged["RS_PTS_AVG"]).fillna(0.0).round(2)
    merged["REB_ELEVATION"] = (merged["PLAYOFF_REB_AVG"] - merged["RS_REB_AVG"]).fillna(0.0).round(2)
    merged["AST_ELEVATION"] = (merged["PLAYOFF_AST_AVG"] - merged["RS_AST_AVG"]).fillna(0.0).round(2)

    result = merged[[
        "PLAYOFF_GAMES", "PLAYOFF_PTS_AVG", "PLAYOFF_REB_AVG", "PLAYOFF_AST_AVG",
        "PTS_ELEVATION", "REB_ELEVATION", "AST_ELEVATION",
    ]]

    logger.info(
        f"Elevation table ready: {len(result)} players | "
        f"avg PTS elevation: {result['PTS_ELEVATION'].mean():+.2f}"
    )
    return result
