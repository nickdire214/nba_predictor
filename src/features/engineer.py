import pandas as pd
import numpy as np
import os
import glob
from loguru import logger


def load_latest_raw(prefix: str, filename: str = None) -> pd.DataFrame:
    """
    Load a raw CSV from data/raw/.
    - If `filename` is given (without extension), load that exact file.
    - Otherwise glob for the most recent prefix_*.csv.
    """
    if filename is not None:
        path = os.path.join("data", "raw", f"{filename}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        logger.info(f"Loading: {path}")
        return pd.read_csv(path)

    pattern = os.path.join("data", "raw", f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    latest = files[-1]
    logger.info(f"Loading: {latest}")
    return pd.read_csv(latest)


def prepare_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and sort the raw player game log DataFrame.
    Must be called before any feature engineering.
    """
    # Parse game date to proper datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Sort each player's games oldest to newest — critical for rolling calcs
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # Parse minutes played — stored as "32:45" string, convert to float
    df["MIN"] = df["MIN"].apply(parse_minutes)

    logger.info(f"Prepared player DataFrame: {df.shape}")
    return df


def parse_minutes(min_str) -> float:
    """Convert '32:45' format to 32.75 decimal minutes."""
    try:
        if pd.isna(min_str):
            return 0.0
        parts = str(min_str).split(":")
        return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# FEATURE GROUP 1 — Rolling averages
# ─────────────────────────────────────────────

def add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average features for key stats over last 5, 10, and 20 games.
    Uses shift(1) to prevent data leakage.
    Builds all columns at once to avoid DataFrame fragmentation.
    """
    logger.info("Adding rolling average features...")

    stats_to_roll = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
                     "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                     "FTM", "FTA", "FT_PCT", "PLUS_MINUS",
                     "OREB", "DREB", "PF", "PFD"]

    windows = [5, 10, 20]
    new_cols = {}

    for stat in stats_to_roll:
        if stat not in df.columns:
            continue
        for window in windows:
            col_name = f"{stat}_avg_L{window}"
            new_cols[col_name] = (
                df.groupby("PLAYER_ID")[stat]
                .transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean())
            )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"Added {len(new_cols)} rolling average columns.")
    return df


# ─────────────────────────────────────────────
# FEATURE GROUP 2 — Rest days
# ─────────────────────────────────────────────

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate days of rest between games for each player.
    """
    logger.info("Adding rest day features...")

    new_cols = {
        "DAYS_REST": (
            df.groupby("PLAYER_ID")["GAME_DATE"]
            .transform(lambda x: x.diff().dt.days - 1)
            .fillna(7).clip(upper=14)
        )
    }
    new_cols["IS_BACK_TO_BACK"] = (new_cols["DAYS_REST"] == 0).astype(int)
    new_cols["IS_WELL_RESTED"] = (new_cols["DAYS_REST"] >= 3).astype(int)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("Added DAYS_REST, IS_BACK_TO_BACK, IS_WELL_RESTED.")
    return df

# ─────────────────────────────────────────────
# FEATURE GROUP 3 — Home / away
# ─────────────────────────────────────────────

def add_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive home/away flag from the MATCHUP column.
    """
    logger.info("Adding home/away features...")

    new_cols = {
        "IS_HOME": df["MATCHUP"].str.contains("vs.").astype(int)
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("Added IS_HOME flag.")
    return df

# ─────────────────────────────────────────────
# FEATURE GROUP 4 — Defensive strength
# ─────────────────────────────────────────────

def add_defensive_strength(player_df: pd.DataFrame, team_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Attach opponent defensive strength to each player game.
    Uses team game logs to calculate how many points each team
    allows on average — then joins that onto the player DataFrame.
    If team_df is not provided, loads the latest single-season file.
    """
    logger.info("Adding defensive strength features...")

    if team_df is None:
        team_df = load_latest_raw("team_gamelogs")
    team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"])

    # For each game, the opponent is the other team in the same GAME_ID
    # We need points ALLOWED — which is the opponent's PTS in that game
    # Merge team_df with itself on GAME_ID to pair up opponents
    home = team_df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]].copy()
    away = team_df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]].copy()

    matchups = home.merge(away, on="GAME_ID", suffixes=("_team", "_opp"))

    # Remove rows where a team is matched with itself
    matchups = matchups[matchups["TEAM_ID_team"] != matchups["TEAM_ID_opp"]]

    # For each team, calculate rolling average points allowed (opponent's PTS)
    matchups = matchups.sort_values(["TEAM_ID_team", "GAME_ID"])
    matchups["OPP_PTS_allowed_avg_L10"] = (
        matchups.groupby("TEAM_ID_team")["PTS_opp"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Keep only what we need for the join
    opp_defense = matchups[["GAME_ID", "TEAM_ID_team", "OPP_PTS_allowed_avg_L10"]].copy()
    opp_defense.columns = ["GAME_ID", "TEAM_ID", "OPP_PTS_allowed_avg_L10"]

    # Join onto player DataFrame — each player inherits their team's opponent defensive rating
    player_df = player_df.merge(opp_defense, on=["GAME_ID", "TEAM_ID"], how="left")

    logger.info("Added OPP_PTS_allowed_avg_L10.")
    return player_df


# ─────────────────────────────────────────────
# FEATURE GROUP 5 — Positional matchup
# ─────────────────────────────────────────────

def add_positional_matchup_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add position-aware defensive strength features.

    Assigns each player a position group (BIG / WING / GUARD) based on
    their rolling averages (shift(1)-lagged -- no leakage):
        BIG   : REB_avg_L10 >= 6.0
        WING  : PTS_avg_L10 >= 10.0 and REB_avg_L10 < 6.0
        GUARD : everything else

    Then computes how many points each team allowed to each position group
    on average over their last 10 games.

    Adds:
        POSITION_GROUP           -- BIG, WING, or GUARD
        OPP_PTS_vs_BIGS_L10     -- rolling L10 pts allowed to bigs
        OPP_PTS_vs_WINGS_L10    -- rolling L10 pts allowed to wings
        OPP_PTS_vs_GUARDS_L10   -- rolling L10 pts allowed to guards
        OPP_PTS_vs_POSITION_L10 -- value for this player's own position group
    """
    logger.info("Adding positional matchup features...")

    # Step 1: assign position group using rolling averages already in df
    reb = player_df["REB_avg_L10"].fillna(0)
    pts = player_df["PTS_avg_L10"].fillna(0)

    player_df = player_df.copy()
    player_df["POSITION_GROUP"] = np.select(
        [reb >= 6.0, (pts >= 10.0) & (reb < 6.0)],
        ["BIG", "WING"],
        default="GUARD",
    )

    # Step 2: parse opponent team ID from MATCHUP
    player_df["_OPP_ABBR"] = player_df["MATCHUP"].apply(
        lambda m: m.split("vs. ")[-1].strip() if "vs." in m else m.split("@ ")[-1].strip()
    )
    abbr_to_id = dict(zip(player_df["TEAM_ABBREVIATION"], player_df["TEAM_ID"]))
    player_df["_OPP_TEAM_ID"] = player_df["_OPP_ABBR"].map(abbr_to_id)

    # Step 3: per game, sum actual PTS scored against each defensive team by position group
    game_pos = (
        player_df.groupby(["GAME_ID", "_OPP_TEAM_ID", "POSITION_GROUP"])["PTS"]
        .sum()
        .reset_index()
        .rename(columns={"_OPP_TEAM_ID": "DEF_TEAM_ID", "PTS": "PTS_SCORED"})
    )

    # Pivot to one row per (GAME_ID, DEF_TEAM_ID) with a column per position
    game_wide = game_pos.pivot_table(
        index=["GAME_ID", "DEF_TEAM_ID"],
        columns="POSITION_GROUP",
        values="PTS_SCORED",
        fill_value=0,
    ).reset_index()
    game_wide.columns.name = None

    for pos in ["BIG", "WING", "GUARD"]:
        if pos not in game_wide.columns:
            game_wide[pos] = 0.0

    # Attach GAME_DATE for correct sort order
    game_dates = player_df[["GAME_ID", "GAME_DATE"]].drop_duplicates("GAME_ID")
    game_wide = game_wide.merge(game_dates, on="GAME_ID", how="left")
    game_wide = game_wide.sort_values(["DEF_TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Step 4: rolling L10 per defensive team per position (shift(1) to avoid leakage)
    for pos, col in [("BIG", "OPP_PTS_vs_BIGS_L10"),
                     ("WING", "OPP_PTS_vs_WINGS_L10"),
                     ("GUARD", "OPP_PTS_vs_GUARDS_L10")]:
        game_wide[col] = (
            game_wide.groupby("DEF_TEAM_ID")[pos]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )

    join_cols = ["GAME_ID", "DEF_TEAM_ID",
                 "OPP_PTS_vs_BIGS_L10", "OPP_PTS_vs_WINGS_L10", "OPP_PTS_vs_GUARDS_L10"]
    opp_pos = game_wide[join_cols].rename(columns={"DEF_TEAM_ID": "_OPP_TEAM_ID"})

    # Step 5: join back onto player df
    player_df = player_df.merge(opp_pos, on=["GAME_ID", "_OPP_TEAM_ID"], how="left")

    # Step 6: each player gets the value for their own position
    player_df["OPP_PTS_vs_POSITION_L10"] = np.select(
        [
            player_df["POSITION_GROUP"] == "BIG",
            player_df["POSITION_GROUP"] == "WING",
            player_df["POSITION_GROUP"] == "GUARD",
        ],
        [
            player_df["OPP_PTS_vs_BIGS_L10"],
            player_df["OPP_PTS_vs_WINGS_L10"],
            player_df["OPP_PTS_vs_GUARDS_L10"],
        ],
        default=np.nan,
    )

    player_df = player_df.drop(columns=["_OPP_ABBR", "_OPP_TEAM_ID"], errors="ignore")

    null_count = int(player_df["OPP_PTS_vs_POSITION_L10"].isna().sum())
    if null_count:
        logger.warning(f"  {null_count} rows have null OPP_PTS_vs_POSITION_L10 after join.")

    pos_counts = player_df["POSITION_GROUP"].value_counts()
    logger.info(
        "Position groups: "
        + ", ".join(f"{k}: {v:,}" for k, v in pos_counts.items())
    )
    logger.info("Added OPP_PTS_vs_BIGS_L10, OPP_PTS_vs_WINGS_L10, "
                "OPP_PTS_vs_GUARDS_L10, OPP_PTS_vs_POSITION_L10.")
    return player_df


# ─────────────────────────────────────────────
# FEATURE GROUP 6 — Pace
# ─────────────────────────────────────────────

def add_pace_features(player_df: pd.DataFrame, team_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add team and opponent pace features to each player game row.

    Pace per game = (FGA + 0.44*FTA - OREB + TOV) normalized to 48 minutes.
    Rolling L10 averages are shift(1)-lagged to prevent data leakage.

    Adds: TEAM_PACE_L10, OPP_PACE_L10, COMBINED_PACE
    """
    logger.info("Adding pace features...")

    if team_df is None:
        team_df = load_latest_raw("team_gamelogs")
    team_df = team_df.copy()
    team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"])

    # Parse team minutes to float (handles "240:00" or plain numeric)
    team_df["MIN_float"] = team_df["MIN"].apply(parse_minutes)
    # Guard against zero minutes
    team_df["MIN_float"] = team_df["MIN_float"].replace(0, np.nan)

    # Raw possessions estimate per game
    team_df["RAW_PACE"] = (
        team_df["FGA"] + 0.44 * team_df["FTA"] - team_df["OREB"] + team_df["TOV"]
    )
    # Normalize to 48 minutes — team MIN is the game duration (~48 for regulation)
    team_df["PACE_48"] = team_df["RAW_PACE"] * (48.0 / team_df["MIN_float"])

    # Sort and compute rolling L10 per team (shift(1) to avoid leakage)
    team_df = team_df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    team_df["TEAM_PACE_L10"] = (
        team_df.groupby("TEAM_ID")["PACE_48"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Self-join to pair each team with its opponent in the same game
    team_pairs = team_df[["GAME_ID", "TEAM_ID", "TEAM_PACE_L10"]].merge(
        team_df[["GAME_ID", "TEAM_ID", "TEAM_PACE_L10"]],
        on="GAME_ID",
        suffixes=("_team", "_opp"),
    )
    team_pairs = team_pairs[team_pairs["TEAM_ID_team"] != team_pairs["TEAM_ID_opp"]]
    team_pairs = team_pairs.rename(columns={
        "TEAM_ID_team":      "TEAM_ID",
        "TEAM_PACE_L10_team": "TEAM_PACE_L10",
        "TEAM_PACE_L10_opp":  "OPP_PACE_L10",
    })[["GAME_ID", "TEAM_ID", "TEAM_PACE_L10", "OPP_PACE_L10"]]

    team_pairs["COMBINED_PACE"] = (team_pairs["TEAM_PACE_L10"] + team_pairs["OPP_PACE_L10"]) / 2.0

    # Join onto player DataFrame on (GAME_ID, TEAM_ID)
    player_df = player_df.merge(team_pairs, on=["GAME_ID", "TEAM_ID"], how="left")

    null_count = player_df["TEAM_PACE_L10"].isna().sum()
    if null_count:
        logger.warning(f"  {null_count} rows have null TEAM_PACE_L10 after join.")

    logger.info("Added TEAM_PACE_L10, OPP_PACE_L10, COMBINED_PACE.")
    return player_df


# ─────────────────────────────────────────────
# FEATURE GROUP 6 — Return from injury
# ─────────────────────────────────────────────

def add_return_from_injury_features(df: pd.DataFrame, return_window: int = 1) -> pd.DataFrame:
    """
    Detect players returning from a multi-game absence (3+ games missed)
    and add a counter tracking how far into their comeback they are.

    Uses DAYS_REST >= 7 as the absence threshold (roughly 3+ games missed
    given the NBA's ~2-day game cadence). Season boundaries are excluded
    so an offseason break never counts as an injury absence.

    Args:
        return_window: number of games after return to flag as IS_RETURNING.
                       1 = first game only, 2 = first two, 3 = first three.

    Adds:
        GAMES_SINCE_RETURN  -- games played since last 3+ game absence;
                               99 = no recent absence (sentinel)
        IS_RETURNING        -- 1 for first `return_window` games back
        RETURN_GAME_NUMBER  -- same as GAMES_SINCE_RETURN capped at 5;
                               NaN when not returning
    """
    logger.info(f"Adding return-from-injury features (return_window={return_window})...")

    # df is already sorted [PLAYER_ID, GAME_DATE] from prepare_player_df
    ABSENCE_DAYS = 7  # >= 7 days gap => missed ~3+ games

    def _per_player(group):
        days_rest  = group["DAYS_REST"].values
        seasons    = group["SEASON_YEAR"].values
        n          = len(days_rest)
        result     = np.full(n, 99.0)

        counter    = 99
        prev_season = None

        for i in range(n):
            curr_season = seasons[i]

            # First game ever, or season boundary: not an injury absence
            if prev_season is None or curr_season != prev_season:
                counter = 99
                result[i] = 99
                prev_season = curr_season
                continue

            prev_season = curr_season
            dr = days_rest[i]

            if pd.isna(dr):
                result[i] = counter
                continue

            if dr >= ABSENCE_DAYS:
                counter = 0
            elif counter < 99:
                counter += 1
            # else stays 99 (no absence has occurred yet)

            result[i] = counter

        return pd.Series(result, index=group.index)

    games_since = (
        df.groupby("PLAYER_ID", group_keys=False)
        .apply(_per_player)
    )

    new_cols = {
        "GAMES_SINCE_RETURN": games_since,
        "IS_RETURNING":       (games_since <= (return_window - 1)).astype(int),
        # NaN for rows where player is not within 6 games of a return
        "RETURN_GAME_NUMBER": games_since.where(games_since <= 5),
    }

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    n_returning = int(new_cols["IS_RETURNING"].sum())
    logger.info(
        f"Added GAMES_SINCE_RETURN, IS_RETURNING, RETURN_GAME_NUMBER. "
        f"{n_returning:,} returning player-games flagged."
    )
    return df


# ─────────────────────────────────────────────
# FEATURE GROUP 7 — Teammate / opponent absence
# ─────────────────────────────────────────────

def add_absence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add teammate usage absorption and opponent absence features.

    KEY_TEAMMATE_OUT (binary) is replaced with:
      TEAMMATE_USAGE_ABSORBED -- estimated extra PTS this player absorbs
                                 when key teammates are missing, based on
                                 their proportional share of team scoring
      TEAMMATES_OUT_COUNT     -- how many key teammates (top-20% avg PTS)
                                 are absent from this game

    KEY_OPP_OUT (binary) is unchanged.

    Vectorized — no row-by-row apply.
    """
    logger.info("Adding absence features...")

    # ── Key player threshold (top 20% by season avg PTS) ─────────────────────
    avg_pts = (
        df.groupby("PLAYER_ID")["PTS"]
        .mean()
        .reset_index()
        .rename(columns={"PTS": "SEASON_AVG_PTS"})
    )
    threshold = avg_pts["SEASON_AVG_PTS"].quantile(0.80)
    key_player_set = set(avg_pts[avg_pts["SEASON_AVG_PTS"] >= threshold]["PLAYER_ID"])
    logger.info(
        f"Key player threshold: {threshold:.1f} avg PTS "
        f"— {len(key_player_set)} players qualify."
    )

    # ── Per-game rolling PTS_avg_L10 for each player (already computed) ───────
    # Use shift(1) value already in df; fill missing with season avg
    df = df.merge(avg_pts, on="PLAYER_ID", how="left")
    pts_l10 = df["PTS_avg_L10"].fillna(df["SEASON_AVG_PTS"]).fillna(0)

    # ── Count key players active per (GAME_ID, TEAM_ID) ─────────────────────
    key_df = df[df["PLAYER_ID"].isin(key_player_set)].copy()
    key_counts = (
        key_df.groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"PLAYER_ID": "KEY_PLAYERS_ACTIVE"})
    )
    max_key = (
        key_counts.groupby("TEAM_ID")["KEY_PLAYERS_ACTIVE"]
        .max()
        .reset_index()
        .rename(columns={"KEY_PLAYERS_ACTIVE": "MAX_KEY_PLAYERS"})
    )
    key_counts = key_counts.merge(max_key, on="TEAM_ID")
    key_counts["TEAMMATES_OUT_COUNT"] = (
        key_counts["MAX_KEY_PLAYERS"] - key_counts["KEY_PLAYERS_ACTIVE"]
    ).clip(lower=0)

    # ── Sum of missing key teammates' avg PTS per (GAME_ID, TEAM_ID) ─────────
    # Active key players in each game
    active_key = key_df[["GAME_ID", "TEAM_ID", "PLAYER_ID"]].copy()
    active_key = active_key.merge(
        df[["PLAYER_ID", "GAME_ID", "PTS_avg_L10", "SEASON_AVG_PTS"]],
        on=["PLAYER_ID", "GAME_ID"],
        how="left",
    )
    active_key["PTS_L10"] = (
        active_key["PTS_avg_L10"].fillna(active_key["SEASON_AVG_PTS"]).fillna(0)
    )
    # All key players' avg PTS summed per team (season-level expected pool)
    key_total_pts = (
        avg_pts[avg_pts["PLAYER_ID"].isin(key_player_set)]
        .merge(
            df[["PLAYER_ID", "TEAM_ID"]].drop_duplicates("PLAYER_ID"),
            on="PLAYER_ID",
            how="left",
        )
        .groupby("TEAM_ID")["SEASON_AVG_PTS"]
        .sum()
        .reset_index()
        .rename(columns={"SEASON_AVG_PTS": "TEAM_KEY_PTS_POOL"})
    )
    # PTS from active key players in each game
    active_sum = (
        active_key.groupby(["GAME_ID", "TEAM_ID"])["PTS_L10"]
        .sum()
        .reset_index()
        .rename(columns={"PTS_L10": "ACTIVE_KEY_PTS"})
    )
    key_counts = (
        key_counts
        .merge(active_sum, on=["GAME_ID", "TEAM_ID"], how="left")
        .merge(key_total_pts, on="TEAM_ID", how="left")
    )
    key_counts["ACTIVE_KEY_PTS"] = key_counts["ACTIVE_KEY_PTS"].fillna(0)
    key_counts["TEAM_KEY_PTS_POOL"] = key_counts["TEAM_KEY_PTS_POOL"].fillna(0)
    # Missing PTS = pool minus active; 0 when no one is missing
    key_counts["MISSING_KEY_PTS"] = (
        key_counts["TEAM_KEY_PTS_POOL"] - key_counts["ACTIVE_KEY_PTS"]
    ).clip(lower=0)

    # ── Join onto player df to compute per-player usage share ────────────────
    df = df.merge(
        key_counts[["GAME_ID", "TEAM_ID", "TEAMMATES_OUT_COUNT", "MISSING_KEY_PTS"]],
        on=["GAME_ID", "TEAM_ID"],
        how="left",
    )
    df["TEAMMATES_OUT_COUNT"] = df["TEAMMATES_OUT_COUNT"].fillna(0)
    df["MISSING_KEY_PTS"]     = df["MISSING_KEY_PTS"].fillna(0)

    # Each player's share of their team's available (non-key) scoring
    # Sum of PTS_avg_L10 for all non-missing players per game/team
    all_avail_pts = (
        df.assign(PTS_L10_fill=pts_l10)
        .groupby(["GAME_ID", "TEAM_ID"])["PTS_L10_fill"]
        .sum()
        .reset_index()
        .rename(columns={"PTS_L10_fill": "TEAM_AVAIL_PTS_SUM"})
    )
    df = df.merge(all_avail_pts, on=["GAME_ID", "TEAM_ID"], how="left")
    df["TEAM_AVAIL_PTS_SUM"] = df["TEAM_AVAIL_PTS_SUM"].replace(0, np.nan)

    # Player scoring share * missing PTS = estimated extra usage
    player_share = pts_l10 / df["TEAM_AVAIL_PTS_SUM"]
    df["TEAMMATE_USAGE_ABSORBED"] = (
        (player_share * df["MISSING_KEY_PTS"]).fillna(0).round(2)
    )

    # ── Star-level teammate absence (20+ PTS_avg_L10) ────────────────────────
    # Higher threshold than the general key player pool — true star absences only
    STAR_THRESHOLD = 20.0
    star_player_set = set(avg_pts[avg_pts["SEASON_AVG_PTS"] >= STAR_THRESHOLD]["PLAYER_ID"])
    logger.info(
        f"Star player threshold: {STAR_THRESHOLD:.0f} avg PTS "
        f"— {len(star_player_set)} players qualify."
    )
    star_df = df[df["PLAYER_ID"].isin(star_player_set)].copy()
    star_active = (
        star_df.groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"PLAYER_ID": "STARS_ACTIVE"})
    )
    max_stars = (
        star_active.groupby("TEAM_ID")["STARS_ACTIVE"]
        .max()
        .reset_index()
        .rename(columns={"STARS_ACTIVE": "MAX_STARS"})
    )
    star_counts = star_active.merge(max_stars, on="TEAM_ID")
    star_counts["KEY_TEAMMATE_OUT"] = (
        star_counts["STARS_ACTIVE"] < star_counts["MAX_STARS"]
    ).astype(int)

    df = df.merge(
        star_counts[["GAME_ID", "TEAM_ID", "KEY_TEAMMATE_OUT"]],
        on=["GAME_ID", "TEAM_ID"],
        how="left",
    )
    # Teams with no stars: flag stays 0 (no star ever present to be absent)
    df["KEY_TEAMMATE_OUT"] = df["KEY_TEAMMATE_OUT"].fillna(0).astype(int)

    # ── Opponent absence (KEY_OPP_OUT unchanged — binary flag) ───────────────
    df["OPP_ABBREVIATION"] = df["MATCHUP"].apply(
        lambda m: m.split("vs. ")[-1].strip() if "vs." in m else m.split("@ ")[-1].strip()
    )
    team_id_lookup = dict(zip(df["TEAM_ABBREVIATION"], df["TEAM_ID"]))
    df["OPP_TEAM_ID"] = df["OPP_ABBREVIATION"].map(team_id_lookup)

    opp_out = key_counts[["GAME_ID", "TEAM_ID", "TEAMMATES_OUT_COUNT"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "TEAMMATES_OUT_COUNT": "_OPP_OUT_COUNT"}
    )
    df = df.merge(opp_out, on=["GAME_ID", "OPP_TEAM_ID"], how="left")
    df["KEY_OPP_OUT"] = (df["_OPP_OUT_COUNT"].fillna(0) > 0).astype(int)

    # ── Clean up helper columns ───────────────────────────────────────────────
    df = df.drop(
        columns=["OPP_ABBREVIATION", "OPP_TEAM_ID", "MISSING_KEY_PTS",
                 "TEAM_AVAIL_PTS_SUM", "SEASON_AVG_PTS", "_OPP_OUT_COUNT"],
        errors="ignore",
    )

    n_absorbed  = int((df["TEAMMATE_USAGE_ABSORBED"] > 0).sum())
    n_star_out  = int(df["KEY_TEAMMATE_OUT"].sum())
    logger.info(
        f"Added TEAMMATE_USAGE_ABSORBED, TEAMMATES_OUT_COUNT, "
        f"KEY_TEAMMATE_OUT, KEY_OPP_OUT. "
        f"{n_absorbed:,} player-games with absorbed usage; "
        f"{n_star_out:,} with a star teammate absent."
    )
    return df

# ─────────────────────────────────────────────
# FEATURE GROUP 6 — Foul features
# ─────────────────────────────────────────────

def add_foul_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add foul-based features normalized to per-36 minutes.
    PF  = fouls committed — high rate = foul trouble risk
    PFD = fouls drawn    — high rate = free throw volume signal
    """
    logger.info("Adding foul features...")

    new_cols = {}

    # Per-36 normalization helper — avoid division by zero
    def per36(stat, minutes):
        return (stat / minutes.replace(0, np.nan)) * 36

    # Rolling per-36 foul rate committed over last 10 games
    # We compute it from the raw rolling averages we already have
    pf_avg  = df.groupby("PLAYER_ID")["PF"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    min_avg = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    pfd_avg = df.groupby("PLAYER_ID")["PFD"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    new_cols["PF_per36_avg_L10"]  = per36(pf_avg, min_avg)
    new_cols["PFD_per36_avg_L10"] = per36(pfd_avg, min_avg)

    # Foul trouble risk flag — above 4.5 fouls per 36 is high risk
    new_cols["FOUL_TROUBLE_RISK"] = (
        new_cols["PF_per36_avg_L10"] > 4.5
    ).astype(int)

    # High free throw draw flag — above 5.0 PFD per 36 is elite drawing ability
    new_cols["HIGH_FT_DRAW"] = (
        new_cols["PFD_per36_avg_L10"] > 5.0
    ).astype(int)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("Added PF_per36_avg_L10, PFD_per36_avg_L10, FOUL_TROUBLE_RISK, HIGH_FT_DRAW.")
    return df
    
# ─────────────────────────────────────────────
# Run all features and save
# ─────────────────────────────────────────────

def build_player_features(use_historical: bool = False, return_window: int = 1) -> pd.DataFrame:
    """
    Master function — loads raw data, runs all feature groups,
    saves to data/features/ and returns the final DataFrame.
    When use_historical=True, loads from the multi-season historical files.
    return_window controls how many games after a return IS_RETURNING stays 1.
    """
    if use_historical:
        df       = load_latest_raw("player_gamelogs", filename="player_gamelogs_historical")
        team_df  = load_latest_raw("team_gamelogs",   filename="team_gamelogs_historical")
    else:
        df       = load_latest_raw("player_gamelogs")
        team_df  = None  # add_defensive_strength will load it internally

    df = prepare_player_df(df)
    df = add_rolling_averages(df)
    df = add_rest_days(df)
    df = add_home_away(df)
    df = add_defensive_strength(df, team_df=team_df)
    df = add_positional_matchup_features(df)
    df = add_pace_features(df, team_df=team_df)
    df = add_return_from_injury_features(df, return_window=return_window)
    df = add_absence_features(df)
    df = add_foul_features(df)

    out_dir = os.path.join("data", "features")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "player_features.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved feature matrix to {out_path} — shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = build_player_features()

    print(f"\nShape: {df.shape}")
    print(f"\nSample — foul features:")
    print(df[["PLAYER_NAME", "GAME_DATE", "PF", "PFD",
              "PF_per36_avg_L10", "PFD_per36_avg_L10",
              "FOUL_TROUBLE_RISK", "HIGH_FT_DRAW"]].head(10).to_string())
