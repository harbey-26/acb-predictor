"""
Main scraping pipeline for ACB basketball data.

Strategy:
  1. Use api2.acb.com (private but public-key API embedded in live.acb.com)
  2. Get all rounds for each target season from competition-data endpoint
  3. For each round, get match list (matchlist endpoint)
  4. For each finished match, get full box scores (boxscores endpoint)
  5. Parse into flat dicts → JSON per match → consolidated CSV

Usage:
    # Quick smoke test with a few known matches
    python -m scraper.pipeline --test

    # Full scrape for all 5 seasons (~2000 matches, ~20-30 min)
    python -m scraper.pipeline

    # Scrape specific seasons only
    python -m scraper.pipeline --seasons 2023-24 2024-25

    # Resume interrupted scrape (skips already-saved matches)
    python -m scraper.pipeline --resume

    # Only merge raw JSONs into processed/matches.csv
    python -m scraper.pipeline --consolidate
"""

import json
import time
import logging
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

from typing import Dict, List

from scraper.acb_api import (
    EDITION_IDS,
    get_all_editions,
    get_matchlist_for_round,
    get_boxscores,
)
from scraper.match_scraper import parse_match
from scraper.season_scanner import TARGET_SEASONS, get_rounds_for_season

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Known match IDs per season for smoke testing
TEST_MATCH_IDS = {
    "2025-26": 104747,  # Unicaja vs Río Breogán, May 2026
    "2023-24": 104000,  # Unicaja vs Lenovo Tenerife, Feb 2024 (Copa del Rey)
    "2020-21": 101262,  # Acunsa GBC vs ?, Sep 2020
}


def _match_path(match_id: int) -> Path:
    return RAW_DIR / f"{match_id}.json"


def already_scraped(match_id: int) -> bool:
    return _match_path(match_id).exists()


def save_match(match_id: int, data: dict) -> None:
    _match_path(match_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_test(session: requests.Session) -> None:
    logger.info("=== TEST MODE (%d matches) ===", len(TEST_MATCH_IDS))

    # Fetch the 2020-21 round 1 to test the full pipeline
    from scraper.acb_api import get_matchlist_for_round
    logger.info("Fetching 2020-21 round 1 matchlist (roundId=5126)…")
    matches = get_matchlist_for_round(session, round_id=5126)
    logger.info("  → %d matches found", len(matches))

    for mi in matches[:3]:
        mid = mi["id"]
        logger.info("  Fetching boxscores for match %d…", mid)
        bs = get_boxscores(session, mid)
        record = parse_match(mi, bs, season="2020-21")
        if record:
            save_match(mid, record)
            logger.info(
                "  ✓ %s vs %s | %d-%d | jornada %s",
                record.get("equipo_local", f'club{record.get("club_local_id")}'),
                record.get("equipo_visitante", f'club{record.get("club_visitante_id")}'),
                record["pts_local"],
                record["pts_visitante"],
                record["jornada"],
            )
        else:
            logger.warning("  ✗ Match %d parse failed", mid)
        time.sleep(0.5)

    logger.info("Test done. Files saved in %s", RAW_DIR)


def scrape_season(
    session: requests.Session,
    season: str,
    all_editions: List[dict],
    resume: bool = False,
    delay: float = 0.4,
) -> int:
    """Scrapes all finished matches for one season. Returns count of scraped matches."""
    rounds = get_rounds_for_season(session, season, all_editions)
    if not rounds:
        logger.warning("No rounds found for season %s", season)
        return 0

    scraped = 0
    errors = 0
    skipped = 0
    match_ids_seen: set[int] = set()

    bar = tqdm(rounds, desc=f"Season {season}", unit="round")

    for round_info in bar:
        round_id = round_info["id"]
        round_num = round_info.get("roundNumber", "?")

        matches = get_matchlist_for_round(session, round_id)
        time.sleep(delay * 0.5)  # brief pause between round fetches

        for match_info in matches:
            mid = match_info["id"]

            if mid in match_ids_seen:
                continue
            match_ids_seen.add(mid)

            if resume and already_scraped(mid):
                skipped += 1
                continue

            if match_info.get("matchStatus") != "FINALIZED":
                continue

            bs = get_boxscores(session, mid)
            record = parse_match(match_info, bs, season=season)

            if record:
                save_match(mid, record)
                scraped += 1
            else:
                errors += 1

            bar.set_postfix(round=round_num, scraped=scraped, skip=skipped, err=errors)
            time.sleep(delay)

    logger.info(
        "Season %s done: %d scraped, %d skipped (resume), %d errors",
        season, scraped, skipped, errors,
    )
    return scraped


def scrape_seasons(
    session: requests.Session,
    seasons: list[str],
    resume: bool = False,
    delay: float = 0.4,
) -> dict[str, int]:
    """Scrapes multiple seasons. Returns {season: count_scraped}."""
    logger.info("Fetching edition metadata (all seasons)…")
    all_editions = get_all_editions(session)
    if not all_editions:
        logger.error("Could not fetch edition data from ACB API")
        return {}

    counts: Dict[str, int] = {}
    for season in seasons:
        logger.info("── Season %s ──────────────────", season)
        counts[season] = scrape_season(session, season, all_editions, resume=resume, delay=delay)

    return counts


def consolidate_to_csv() -> Path:
    """Merges all raw JSON match files into data/processed/matches.csv."""
    import pandas as pd

    processed_dir = RAW_DIR.parent / "processed"
    processed_dir.mkdir(exist_ok=True)

    records = []
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            pass

    if not records:
        logger.warning("No JSON files to consolidate.")
        return processed_dir / "matches.csv"

    df = pd.DataFrame(records)

    # Ensure consistent column order
    id_cols = ["match_id", "temporada", "fecha", "jornada", "round_type", "round_id",
               "week_id", "equipo_local", "equipo_visitante",
               "club_local_id", "club_visitante_id",
               "pts_local", "pts_visitante", "ganador"]
    stat_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + sorted(stat_cols)]

    out_path = processed_dir / "matches.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Consolidated %d matches → %s", len(df), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="ACB basketball data scraper")
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test: scrape first 3 matches of 2020-21 round 1"
    )
    parser.add_argument(
        "--seasons", nargs="+", default=TARGET_SEASONS,
        metavar="SEASON",
        help=f"Seasons to scrape (default: all). Available: {TARGET_SEASONS}"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-downloaded matches"
    )
    parser.add_argument(
        "--delay", type=float, default=0.4,
        help="Seconds between requests (default: 0.4)"
    )
    parser.add_argument(
        "--consolidate", action="store_true",
        help="Only merge raw JSONs into processed/matches.csv"
    )
    args = parser.parse_args()

    session = requests.Session()

    if args.consolidate:
        consolidate_to_csv()
        return

    if args.test:
        run_test(session)
        return

    invalid = [s for s in args.seasons if s not in TARGET_SEASONS]
    if invalid:
        parser.error(f"Unknown seasons: {invalid}. Valid: {TARGET_SEASONS}")

    counts = scrape_seasons(session, args.seasons, resume=args.resume, delay=args.delay)

    total = sum(counts.values())
    logger.info("══ DONE: %d matches total ══", total)
    for s, n in counts.items():
        logger.info("  %s → %d matches", s, n)

    logger.info("Consolidating to CSV…")
    consolidate_to_csv()


if __name__ == "__main__":
    main()
