import unicodedata
import urllib.request
import json
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)

# ESPN uses full team names; map to standard NBA abbreviations
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "LA Clippers":            "LAC",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")


def fetch_injury_report() -> pd.DataFrame:
    """
    Fetch the current NBA injury report from the ESPN unofficial API.

    Returns a DataFrame with columns:
        PLAYER_NAME, TEAM_ABBREVIATION, STATUS, DESCRIPTION, UPDATED

    STATUS values: "Out", "Questionable", "Day-To-Day", "Probable"
    """
    logger.info(f"Fetching injury report from ESPN API...")

    try:
        req = urllib.request.Request(
            ESPN_INJURIES_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except Exception as e:
        logger.error(f"Failed to fetch ESPN injury report: {e}")
        raise

    rows = []
    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        team_abbr = TEAM_NAME_TO_ABBR.get(team_name, team_name[:3].upper())

        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            player_name = _ascii(athlete.get("displayName", ""))

            # Parse ISO timestamp to a readable string
            raw_date = inj.get("date", "")
            try:
                dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                updated = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                updated = raw_date

            rows.append({
                "PLAYER_NAME":       player_name,
                "TEAM_ABBREVIATION": team_abbr,
                "STATUS":            inj.get("status", ""),
                "DESCRIPTION":       inj.get("shortComment", inj.get("longComment", "")),
                "UPDATED":           updated,
            })

    df = pd.DataFrame(rows, columns=["PLAYER_NAME", "TEAM_ABBREVIATION", "STATUS", "DESCRIPTION", "UPDATED"])
    logger.info(
        f"Fetched {len(df)} injury entries across "
        f"{df['TEAM_ABBREVIATION'].nunique()} teams. "
        f"Status breakdown: {df['STATUS'].value_counts().to_dict()}"
    )
    return df


def get_unavailable_players() -> set:
    """
    Return a set of player names (ASCII) whose STATUS is 'Out'.
    Used to filter predictions before display.
    """
    df = fetch_injury_report()
    out_df = df[df["STATUS"] == "Out"]
    unavailable = set(out_df["PLAYER_NAME"].tolist())
    logger.info(f"Players ruled Out: {len(unavailable)}")
    return unavailable
