import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

from nba_api.stats.endpoints import playergamelogs, teamgamelogs

load_dotenv()


def fetch_player_gamelogs(season: str = "2025-26", season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch every player game log for a given season.
    Each row = one player's stats for one game.
    """
    logger.info(f"Fetching player game logs — season: {season} | type: {season_type}")

    try:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
        )
        df = logs.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed to fetch player game logs: {e}")
        raise

    if df.empty:
        logger.warning("No player game log data returned — season may not have started.")
        return df

    logger.info(f"Fetched {len(df):,} rows across {df['PLAYER_NAME'].nunique():,} players.")
    return df


def fetch_team_gamelogs(season: str = "2025-26", season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch every team game log for a given season.
    Each row = one team's stats for one game.
    """
    logger.info(f"Fetching team game logs — season: {season} | type: {season_type}")

    try:
        logs = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
        )
        df = logs.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed to fetch team game logs: {e}")
        raise

    if df.empty:
        logger.warning("No team game log data returned.")
        return df

    logger.info(f"Fetched {len(df):,} rows across {df['TEAM_NAME'].nunique():,} teams.")
    return df


def fetch_player_gamelogs_multiyear(seasons: list, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch player game logs for multiple seasons and concatenate.
    Returns a single DataFrame with all seasons combined.
    """
    frames = []
    for season in seasons:
        df = fetch_player_gamelogs(season=season, season_type=season_type)
        if not df.empty:
            frames.append(df)
            logger.info(f"  {season}: {len(df):,} rows")
        else:
            logger.warning(f"  {season}: no data returned, skipping.")

    if not frames:
        logger.warning("No data returned for any season.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined player logs: {len(combined):,} rows across {len(frames)} seasons.")
    return combined


def fetch_team_gamelogs_multiyear(seasons: list, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch team game logs for multiple seasons and concatenate.
    Returns a single DataFrame with all seasons combined.
    """
    frames = []
    for season in seasons:
        df = fetch_team_gamelogs(season=season, season_type=season_type)
        if not df.empty:
            frames.append(df)
            logger.info(f"  {season}: {len(df):,} rows")
        else:
            logger.warning(f"  {season}: no data returned, skipping.")

    if not frames:
        logger.warning("No data returned for any season.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined team logs: {len(combined):,} rows across {len(frames)} seasons.")
    return combined


def fetch_playoff_gamelogs(seasons: list = None) -> pd.DataFrame:
    """
    Fetch player game logs for Playoffs across multiple seasons.
    Defaults to the last three completed seasons.
    """
    if seasons is None:
        seasons = ["2022-23", "2023-24", "2024-25"]
    return fetch_player_gamelogs_multiyear(seasons, season_type="Playoffs")


def save_raw(df: pd.DataFrame, filename_prefix: str, season: str = "2025-26") -> str:
    """
    Save a DataFrame to data/raw/ as a timestamped CSV.
    Returns the full file path.
    """
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    season_slug = season.replace("-", "_")
    filename = f"{filename_prefix}_{season_slug}_{timestamp}.csv"
    filepath = os.path.join(raw_dir, filename)

    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df):,} rows to {filepath}")

    return filepath


if __name__ == "__main__":
    # --- Player logs ---
    player_df = fetch_player_gamelogs(season="2025-26")
    player_path = save_raw(player_df, filename_prefix="player_gamelogs", season="2025-26")

    print(f"\n--- Player Game Logs ---")
    print(f"Shape:   {player_df.shape}")
    print(f"Columns: {list(player_df.columns)}")
    print(f"\nSample:\n{player_df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_DATE', 'PTS', 'REB', 'AST']].head(5)}")

    # --- Team logs ---
    team_df = fetch_team_gamelogs(season="2025-26")
    team_path = save_raw(team_df, filename_prefix="team_gamelogs", season="2025-26")

    print(f"\n--- Team Game Logs ---")
    print(f"Shape:   {team_df.shape}")
    print(f"\nSample:\n{team_df[['TEAM_NAME', 'GAME_DATE', 'PTS', 'REB', 'AST', 'WL']].head(5)}")