import os
import json
import unicodedata
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from dotenv import load_dotenv

import pandas as pd
from loguru import logger

load_dotenv()

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
ODDS_URL   = "https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"

EMPTY_COLS = ["PLAYER_NAME", "VEGAS_PTS", "VEGAS_REB", "VEGAS_AST"]


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")


def _get(url: str, params: dict) -> dict:
    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def fetch_player_props(sport: str = "basketball_nba") -> pd.DataFrame:
    """
    Fetch player prop lines from the Odds API for today's NBA games.

    Steps:
      1. GET /v4/sports/basketball_nba/events -> list of today's game event IDs
      2. For each event, GET /events/{id}/odds with player_points, player_rebounds,
         player_assists markets (American odds format, US region)
      3. Parse the over line for each player/market as the Vegas projection
      4. Pivot to one row per player: PLAYER_NAME, VEGAS_PTS, VEGAS_REB, VEGAS_AST

    Returns a DataFrame with columns PLAYER_NAME, VEGAS_PTS, VEGAS_REB, VEGAS_AST.
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise EnvironmentError("ODDS_API_KEY not set in environment / .env")

    # ── Step 1: fetch today's events ───────────────────────────────────────
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    tomorrow = datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59Z")

    logger.info("Fetching NBA events from Odds API...")
    try:
        events = _get(EVENTS_URL, {
            "apiKey":        api_key,
            "sport":         sport,
            "commenceTimeFrom": today,
            "commenceTimeTo":   tomorrow,
        })
    except Exception as e:
        logger.error(f"Failed to fetch events: {e}")
        raise

    if not events:
        logger.warning("No NBA events found for today.")
        return pd.DataFrame(columns=EMPTY_COLS)

    logger.info(f"Found {len(events)} games today.")

    # ── Step 2 & 3: fetch props for each game ──────────────────────────────
    # Accumulate rows: {player_name: {stat: line}}
    player_lines: dict[str, dict] = {}

    MARKET_TO_COL = {
        "player_points":   "VEGAS_PTS",
        "player_rebounds":  "VEGAS_REB",
        "player_assists":   "VEGAS_AST",
    }

    for event in events:
        event_id   = event["id"]
        home_team  = event.get("home_team", "?")
        away_team  = event.get("away_team", "?")
        logger.info(f"  Fetching props for {away_team} @ {home_team} (id={event_id})...")

        try:
            data = _get(
                ODDS_URL.format(event_id=event_id),
                {
                    "apiKey":      api_key,
                    "regions":     "us",
                    "markets":     "player_points,player_rebounds,player_assists",
                    "oddsFormat":  "american",
                },
            )
        except Exception as e:
            logger.warning(f"    Props fetch failed for event {event_id}: {e}")
            continue

        # data is a dict with "bookmakers" list
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            logger.warning(f"    No bookmakers returned for event {event_id}.")
            continue

        # Use the first bookmaker that has all three markets
        chosen_bk = None
        for bk in bookmakers:
            markets_present = {m["key"] for m in bk.get("markets", [])}
            if markets_present >= set(MARKET_TO_COL.keys()):
                chosen_bk = bk
                break
        if chosen_bk is None:
            # Fall back to whichever bookmaker has the most markets
            chosen_bk = max(bookmakers, key=lambda b: len(b.get("markets", [])))

        logger.info(f"    Using bookmaker: {chosen_bk.get('title', 'unknown')}")

        for market in chosen_bk.get("markets", []):
            market_key = market.get("key")
            col = MARKET_TO_COL.get(market_key)
            if col is None:
                continue

            for outcome in market.get("outcomes", []):
                # Each outcome: {"name": "Over"/"Under", "description": player name, "point": 24.5}
                if outcome.get("name") != "Over":
                    continue
                raw_name = outcome.get("description", "")
                if not raw_name:
                    continue
                player = _ascii(raw_name)
                point  = outcome.get("point")
                if point is None:
                    continue

                if player not in player_lines:
                    player_lines[player] = {}
                player_lines[player][col] = float(point)

    if not player_lines:
        logger.warning("No player prop lines parsed from any game.")
        return pd.DataFrame(columns=EMPTY_COLS)

    # ── Step 4: build output DataFrame ────────────────────────────────────
    rows = []
    for player, lines in player_lines.items():
        rows.append({
            "PLAYER_NAME": player,
            "VEGAS_PTS":   lines.get("VEGAS_PTS"),
            "VEGAS_REB":   lines.get("VEGAS_REB"),
            "VEGAS_AST":   lines.get("VEGAS_AST"),
        })

    df = pd.DataFrame(rows, columns=EMPTY_COLS)
    df = df.sort_values("PLAYER_NAME").reset_index(drop=True)

    pts_count = df["VEGAS_PTS"].notna().sum()
    reb_count = df["VEGAS_REB"].notna().sum()
    ast_count = df["VEGAS_AST"].notna().sum()
    logger.info(
        f"Vegas lines parsed: {len(df)} players | "
        f"PTS: {pts_count}, REB: {reb_count}, AST: {ast_count}"
    )
    return df


def get_vegas_lines() -> pd.DataFrame:
    """
    Return Vegas player prop lines, or an empty DataFrame if the fetch fails.
    """
    try:
        return fetch_player_props()
    except Exception as e:
        logger.warning(f"Vegas lines unavailable: {e}")
        return pd.DataFrame(columns=EMPTY_COLS)
