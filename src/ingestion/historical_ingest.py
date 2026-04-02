import os
import sys
import time
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ingestion.nba_stats import (
    fetch_player_gamelogs_multiyear,
    fetch_team_gamelogs_multiyear,
)

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

OUT_DIR = os.path.join("data", "raw")
PLAYER_OUT = os.path.join(OUT_DIR, "player_gamelogs_historical.csv")
TEAM_OUT   = os.path.join(OUT_DIR, "team_gamelogs_historical.csv")


def print_season_breakdown(df: pd.DataFrame, label: str):
    print(f"\n  {label} rows per season:")
    if "SEASON_YEAR" in df.columns:
        season_col = "SEASON_YEAR"
    elif "SEASON_ID" in df.columns:
        season_col = "SEASON_ID"
    else:
        print("  (no season column found for breakdown)")
        return
    for season, count in df[season_col].value_counts().sort_index().items():
        print(f"    {season}: {count:,}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Historical ingest: player game logs")
    print(f"  Seasons: {SEASONS}")
    print("=" * 60)

    t0 = time.time()
    player_df = fetch_player_gamelogs_multiyear(SEASONS)
    player_elapsed = time.time() - t0

    if player_df.empty:
        print("ERROR: No player data returned.")
    else:
        print_season_breakdown(player_df, "Player")
        print(f"\n  Combined shape: {player_df.shape[0]:,} rows x {player_df.shape[1]} cols")
        print(f"  Elapsed: {player_elapsed:.1f}s")
        player_df.to_csv(PLAYER_OUT, index=False)
        print(f"  Saved -> {PLAYER_OUT}")

    print()
    print("=" * 60)
    print("  Historical ingest: team game logs")
    print(f"  Seasons: {SEASONS}")
    print("=" * 60)

    t0 = time.time()
    team_df = fetch_team_gamelogs_multiyear(SEASONS)
    team_elapsed = time.time() - t0

    if team_df.empty:
        print("ERROR: No team data returned.")
    else:
        print_season_breakdown(team_df, "Team")
        print(f"\n  Combined shape: {team_df.shape[0]:,} rows x {team_df.shape[1]} cols")
        print(f"  Elapsed: {team_elapsed:.1f}s")
        team_df.to_csv(TEAM_OUT, index=False)
        print(f"  Saved -> {TEAM_OUT}")

    print()
    print("=" * 60)
    print("  Done")
    print("=" * 60)
