"""
Resolves season names to edition IDs and round lists using the ACB API.
This replaces the sequential ID scanning approach with a cleaner API-first strategy.
"""

import logging
import requests
from typing import Optional, List
from scraper.acb_api import EDITION_IDS, get_all_editions, get_rounds_for_edition

logger = logging.getLogger(__name__)

TARGET_SEASONS = list(EDITION_IDS.keys())


def get_rounds_for_season(
    session: requests.Session,
    season: str,
    all_editions: Optional[List[dict]] = None,
) -> list[dict]:
    """
    Returns the list of round dicts for a season string like "2021-22".
    Each round dict has: {id, roundNumber, subphaseId, subphaseNumber}.
    """
    edition_id = EDITION_IDS.get(season)
    if not edition_id:
        logger.error("Unknown season: %s. Valid: %s", season, TARGET_SEASONS)
        return []

    if all_editions is None:
        all_editions = get_all_editions(session)

    rounds = get_rounds_for_edition(all_editions, edition_id)
    logger.info("Season %s (edition %d): %d rounds", season, edition_id, len(rounds))
    return rounds
