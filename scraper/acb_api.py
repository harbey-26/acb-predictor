"""
Client for the private api2.acb.com REST API used by live.acb.com.

API key is a public static string embedded in the live.acb.com JavaScript bundle.
All endpoints require the X-APIKEY header; no OAuth or user credentials needed.
"""

import time
import logging
import requests
from typing import Optional, List

logger = logging.getLogger(__name__)

BASE = "https://api2.acb.com"
API_KEY = "0dd94928-6f57-4c08-a3bd-b1b2f092976e"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "X-APIKEY": API_KEY,
    "Origin": "https://live.acb.com",
    "Referer": "https://live.acb.com/",
}

# Liga Nacional de Baloncesto = competitionId 1
COMPETITION_ID = 1

# Editions that correspond to each target season (verified via competition-data)
EDITION_IDS = {
    "2020-21": 85,
    "2021-22": 86,
    "2022-23": 87,
    "2023-24": 88,
    "2024-25": 89,
}


def _get(session: requests.Session, path: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    """
    Performs a GET request to the ACB API with retry logic.
    Returns parsed JSON or None on failure.
    """
    url = BASE + path
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
            if resp.status_code == 404:
                return None
            if resp.status_code == 204:
                return None  # No content (match stats not yet available)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("ACB API %s → HTTP %d (attempt %d)", path, resp.status_code, attempt + 1)
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            logger.warning("Timeout on %s (attempt %d)", path, attempt + 1)
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error on %s: %s", path, exc)
            time.sleep(2 ** attempt)
    return None


def get_all_editions(session: requests.Session) -> List[dict]:
    """
    Returns all editions (seasons) for competition 1 (Liga Nacional).
    Each edition has: id, seasonStartYear, seasonEndYear, rounds (list of roundIds).
    """
    data = _get(session, "/api/matchdata/Menu/competition-data", {"competitionIds": COMPETITION_ID})
    if not data:
        return []
    competitions = data.get("competitions", [])
    liga = next((c for c in competitions if c["id"] == COMPETITION_ID), None)
    return liga["editions"] if liga else []


def get_rounds_for_edition(editions: List[dict], edition_id: int) -> List[dict]:
    """Returns the list of rounds for a specific edition."""
    ed = next((e for e in editions if e["id"] == edition_id), None)
    if not ed:
        return []
    return ed.get("rounds", [])


def get_matchlist_for_round(session: requests.Session, round_id: int) -> List[dict]:
    """
    Returns the list of matches for a given round.
    Each match contains: id, startDateTime, matchStatus, homeClubId, awayClubId,
    homeScore, awayScore, roundNumber, roundId, weekId, roundType.
    """
    data = _get(session, "/api/matchdata/Menu/matchlist", {"roundId": round_id})
    if not data:
        return []
    return [m for m in data.get("matches", []) if m.get("startDateTime")]


def get_boxscores(session: requests.Session, match_id: int) -> Optional[dict]:
    """
    Returns full box score data for a match.
    Structure: {matchFinished, teamBoxscores: [{team, headCoach, statsByPeriods}, ...]}
    teamBoxscores[0] = home team, teamBoxscores[1] = away team.
    statsByPeriods[i] where quarter=0 contains full-game totals.
    """
    return _get(session, "/api/matchdata/Result/boxscores", {"matchId": match_id})


def get_match_header(session: requests.Session, match_id: int) -> Optional[dict]:
    """
    Returns match header (metadata): teams, score, date, quarter scores.
    """
    return _get(session, "/api/matchdata/MatchHeader/match-header", {"matchId": match_id})
