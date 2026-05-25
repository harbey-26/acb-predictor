"""
Parses ACB API responses into a flat dictionary suitable for CSV storage.

Data comes from two endpoints:
  - matchlist (round-level): date, teams, score, round info
  - boxscores (match-level): per-player and team aggregate stats
"""

from typing import Optional


def _safe_int(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def _safe_pct(made: int, attempted: int) -> float:
    return round(made / attempted, 4) if attempted > 0 else 0.0


def _extract_team_totals(statsByPeriods: list) -> dict:
    """Extracts full-game (quarter=0) totals for a team."""
    full_game = next((p for p in statsByPeriods if p.get("quarter") == 0), None)
    if not full_game:
        return {}

    players = full_game.get("stats", {}).get("players", [])
    totals: dict = {
        "t2_anotados": 0, "t2_intentados": 0,
        "t3_anotados": 0, "t3_intentados": 0,
        "tl_anotados": 0, "tl_intentados": 0,
        "reb_ofensivos": 0, "reb_defensivos": 0, "reb_totales": 0,
        "asistencias": 0, "perdidas": 0, "recuperaciones": 0,
        "tapones_favor": 0, "tapones_contra": 0,
        "faltas_cometidas": 0, "valoracion": 0,
    }

    for p in players:
        totals["t2_anotados"]      += _safe_int(p.get("twoPointersMade"))
        totals["t2_intentados"]    += _safe_int(p.get("twoPointersAttempted"))
        totals["t3_anotados"]      += _safe_int(p.get("threePointersMade"))
        totals["t3_intentados"]    += _safe_int(p.get("threePointersAttempted"))
        totals["tl_anotados"]      += _safe_int(p.get("freeThrowsMade"))
        totals["tl_intentados"]    += _safe_int(p.get("freeThrowsAttempted"))
        totals["reb_ofensivos"]    += _safe_int(p.get("offRebounds"))
        totals["reb_defensivos"]   += _safe_int(p.get("defRebounds"))
        totals["reb_totales"]      += _safe_int(p.get("totalRebounds"))
        totals["asistencias"]      += _safe_int(p.get("assists"))
        totals["perdidas"]         += _safe_int(p.get("turnovers"))
        totals["recuperaciones"]   += _safe_int(p.get("steals"))
        totals["tapones_favor"]    += _safe_int(p.get("blocks"))
        totals["tapones_contra"]   += _safe_int(p.get("receivedBlocks"))
        totals["faltas_cometidas"] += _safe_int(p.get("personalFouls"))
        totals["valoracion"]       += _safe_int(p.get("rating"))

    # Derived shooting percentages
    totals["t2_pct"] = _safe_pct(totals["t2_anotados"], totals["t2_intentados"])
    totals["t3_pct"] = _safe_pct(totals["t3_anotados"], totals["t3_intentados"])
    totals["tl_pct"] = _safe_pct(totals["tl_anotados"], totals["tl_intentados"])

    return totals


def parse_match(
    match_info: dict,
    boxscores: Optional[dict],
    season: str,
) -> Optional[dict]:
    """
    Combines matchlist row + boxscores into a single flat record.

    match_info: one entry from get_matchlist_for_round()
    boxscores:  result of get_boxscores() — may be None for unplayed matches
    season:     e.g. "2021-22"
    """
    if not match_info:
        return None

    match_id = match_info.get("id")
    status = match_info.get("matchStatus", "")

    # Only process finished matches
    if status != "FINALIZED":
        return None

    home_score = _safe_int(match_info.get("homeScore"))
    away_score = _safe_int(match_info.get("awayScore"))

    if home_score == 0 and away_score == 0:
        return None

    record: dict = {
        "match_id": match_id,
        "temporada": season,
        "fecha": match_info.get("startDateTime", ""),
        "jornada": _safe_int(match_info.get("roundNumber")),
        "round_type": match_info.get("roundType", ""),  # "LR"=liga regular, "PO"=playoffs
        "round_id": match_info.get("roundId"),
        "week_id": match_info.get("weekId"),
        "club_local_id": match_info.get("homeClubId"),
        "club_visitante_id": match_info.get("awayClubId"),
        "pts_local": home_score,
        "pts_visitante": away_score,
        "ganador": "local" if home_score > away_score else "visitante",
    }

    # --- Box score stats ---
    if boxscores and boxscores.get("matchFinished"):
        team_boxes = boxscores.get("teamBoxscores", [])
        if len(team_boxes) >= 2:
            home_box = team_boxes[0]
            away_box = team_boxes[1]

            record["equipo_local"]     = home_box.get("team", {}).get("fullName", "")
            record["equipo_visitante"] = away_box.get("team", {}).get("fullName", "")
            record["entrenador_local"] = home_box.get("headCoach", "")
            record["entrenador_visitante"] = away_box.get("headCoach", "")

            for prefix, box in [("loc", home_box), ("vis", away_box)]:
                stats = _extract_team_totals(box.get("statsByPeriods", []))
                for key, val in stats.items():
                    record[f"{prefix}_{key}"] = val
        else:
            # Boxscores incomplete — keep basic match info
            record["equipo_local"] = ""
            record["equipo_visitante"] = ""
    else:
        record["equipo_local"] = ""
        record["equipo_visitante"] = ""

    return record
