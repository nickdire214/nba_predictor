import pandas as pd
import os
import glob
from loguru import logger
from nba_api.stats.endpoints import commonteamroster


def get_team_rosters(team_ids: list, season: str = "2025-26") -> set:
    """
    Fetch active roster for each team playing tonight.
    Returns a set of player IDs currently on those rosters.
    """
    active_player_ids = set()

    for team_id in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season
            )
            roster_df = roster.get_data_frames()[0]
            ids = roster_df["PLAYER_ID"].tolist()
            active_player_ids.update(ids)
            logger.info(f"Team {team_id} — {len(ids)} players on roster.")
        except Exception as e:
            logger.warning(f"Could not fetch roster for team {team_id}: {e}")

    logger.info(f"Total rostered players across all teams tonight: {len(active_player_ids)}")
    return active_player_ids


def get_recently_active_players(team_ids: list, n_games: int = 3) -> set:
    """
    Check which players have actually appeared in their team's
    last N games using our raw game log data.

    A player who hasn't appeared in the last 3 team games is
    flagged as likely inactive — injured, IR, or load managed.
    Also supplements with current-season playoff logs so that
    playoff players are not incorrectly marked inactive when the
    feature matrix ends at the regular season.
    Returns a set of player IDs who have appeared recently.
    """
    pattern = os.path.join("data", "raw", "player_gamelogs_*.csv")
    files = sorted(glob.glob(pattern))

    recently_active = set()

    if not files:
        logger.warning("No player game log files found — skipping feature-matrix activity check.")
    else:
        df = pd.read_csv(files[-1])
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df[df["TEAM_ID"].isin(team_ids)]

        for team_id in team_ids:
            team_df = df[df["TEAM_ID"] == team_id].sort_values("GAME_DATE")
            recent_dates = team_df["GAME_DATE"].unique()[-n_games:]
            recent_games = team_df[team_df["GAME_DATE"].isin(recent_dates)]
            recently_active.update(recent_games["PLAYER_ID"].tolist())

    n_from_matrix = len(recently_active)
    logger.info(f"Players active from feature matrix: {n_from_matrix}")

    # Supplement with current-season playoff logs so playoff players
    # who last appeared after the feature matrix cutoff are not dropped.
    try:
        from src.ingestion.nba_stats import fetch_player_gamelogs
        playoff_logs = fetch_player_gamelogs(season="2025-26", season_type="Playoffs")
        if not playoff_logs.empty:
            po_ids = set(
                playoff_logs[playoff_logs["TEAM_ID"].isin(team_ids)]["PLAYER_ID"].tolist()
            )
            new_ids = po_ids - recently_active
            recently_active.update(po_ids)
            logger.info(f"Players added from 2025-26 playoff logs: {len(new_ids)}")
        else:
            logger.info("Players added from 2025-26 playoff logs: 0 (no data)")
    except Exception as e:
        logger.warning(f"Could not fetch playoff logs for activity check: {e}")

    logger.info(f"Total active players: {len(recently_active)}")
    return recently_active


def get_available_players(
    team_ids: list,
    season: str = "2025-26",
    n_games: int = 3
) -> set:
    """
    Master function — returns player IDs who are:
    1. On the active roster (not traded/waived)
    2. Have appeared in the last N team games (not injured/IR)

    The intersection of both sets gives us the most reliable
    estimate of who is available to play tonight.
    """
    rostered = get_team_rosters(team_ids, season=season)
    recently_active = get_recently_active_players(team_ids, n_games=n_games)

    if not rostered:
        logger.warning("Roster fetch failed — falling back to activity check only.")
        return recently_active

    if not recently_active:
        logger.warning("Activity check failed — falling back to roster only.")
        return rostered

    # Intersection — must be on roster AND have played recently
    available = rostered & recently_active
    inactive = rostered - recently_active

    if inactive:
        logger.info(f"Flagged {len(inactive)} rostered players as likely inactive "
                    f"(no appearance in last {n_games} games).")

    logger.info(f"Available players for tonight: {len(available)}")
    return available