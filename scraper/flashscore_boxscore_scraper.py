"""
Fase 2: Boxscores BBL desde FlashScore.

Fuente: feed directo df_st_1_{match_id} (global.flashscore.ninja)
No requiere navegador. Devuelve estadísticas completas de equipo:
  FGA, FGM, FG%, 2PA, 2PM, 2P%, 3PA, 3PM, 3P%, FTA, FTM, FT%,
  OREB, DREB, REB, AST, BLK, TOV, STL, PF

Salida: data/flashscore/boxscores/{season}_boxscores.jsonl
  Una línea JSON por partido. Si no hay stats: scrape_status="no_stats".

Uso:
    python -m scraper.flashscore_boxscore_scraper                     # todas
    python -m scraper.flashscore_boxscore_scraper --season 2024-25    # una
    python -m scraper.flashscore_boxscore_scraper --season 2024-25 --resume
    python -m scraper.flashscore_boxscore_scraper --delay 2.0         # pausa entre requests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set

def _print(*args, **kwargs):
    """print() with immediate flush for background-process-safe output."""
    print(*args, **kwargs, flush=True)


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR  = BASE_DIR / "data" / "flashscore" / "index"
BOX_DIR    = BASE_DIR / "data" / "flashscore" / "boxscores"

FEED_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.flashscore.com/",
    "Origin":  "https://www.flashscore.com",
    "X-Fsign": "SW9D1eZo",
}
FEED_BASE = "https://global.flashscore.ninja/2/x/feed/df_st_1_{mid}"

# Mapping: stat label in feed → (home_col, away_col)
STAT_MAP: Dict[str, tuple] = {
    "Field goals attempts":           ("home_fga",   "away_fga"),
    "Field goals made":               ("home_fgm",   "away_fgm"),
    "Field goals %":                  ("home_fg_pct", "away_fg_pct"),
    "2-point field goals attempts":   ("home_fg2a",  "away_fg2a"),
    "2-point field goals made":       ("home_fg2m",  "away_fg2m"),
    "2-point field goals %":          ("home_fg2_pct", "away_fg2_pct"),
    "3-point field goals attempts":   ("home_fg3a",  "away_fg3a"),
    "3-point field goals made":       ("home_fg3m",  "away_fg3m"),
    "3-point field goals %":          ("home_fg3_pct", "away_fg3_pct"),
    "Free throws attempts":           ("home_fta",   "away_fta"),
    "Free throws made":               ("home_ftm",   "away_ftm"),
    "Free throws %":                  ("home_ft_pct", "away_ft_pct"),
    "Offensive rebounds":             ("home_oreb",  "away_oreb"),
    "Defensive rebounds":             ("home_dreb",  "away_dreb"),
    "Total rebounds":                 ("home_reb",   "away_reb"),
    "Assists":                        ("home_ast",   "away_ast"),
    "Blocks":                         ("home_blk",   "away_blk"),
    "Turnovers":                      ("home_tov",   "away_tov"),
    "Steals":                         ("home_stl",   "away_stl"),
    "Personal fouls":                 ("home_pf",    "away_pf"),
}


def _parse_pct(val: str) -> Optional[float]:
    """'46.55%' → 46.55  |  '0' / '' / 'N/A' → None"""
    val = val.strip().rstrip("%")
    try:
        return float(val)
    except ValueError:
        return None


def _parse_int(val: str) -> Optional[int]:
    try:
        return int(val.strip())
    except ValueError:
        return None


def parse_stats_feed(data: str) -> Optional[Dict]:
    """
    Parsea el formato SA÷ del feed df_st_1.
    Extrae solo la sección SE÷Match (totales del partido completo).
    Retorna dict con columnas home_*/away_* o None si no hay datos.
    """
    if not data or "Field goals" not in data:
        return None

    # Extraer sección SE÷Match (hasta el siguiente SE÷ o fin)
    match_section_m = re.search(r"SE÷Match¬~(.*?)(?=SE÷|\Z)", data, re.DOTALL)
    if not match_section_m:
        return None
    section = match_section_m.group(1)

    result: Dict = {}
    # Buscar cada estadística: SG÷{name}¬SH÷{home}¬SI÷{away}¬~
    for name, (home_col, away_col) in STAT_MAP.items():
        # Escapar caracteres especiales en el nombre para regex
        escaped = re.escape(name)
        m = re.search(rf"SG÷{escaped}¬SH÷([^¬]*)¬SI÷([^¬]*)", section)
        if m:
            h_raw, a_raw = m.group(1), m.group(2)
            if "%" in name:
                result[home_col] = _parse_pct(h_raw)
                result[away_col] = _parse_pct(a_raw)
            else:
                result[home_col] = _parse_int(h_raw)
                result[away_col] = _parse_int(a_raw)
        else:
            result[home_col] = None
            result[away_col] = None

    # Verificar que al menos FGA tiene datos
    if result.get("home_fga") is None and result.get("away_fga") is None:
        return None

    return result


def fetch_match_stats(match_id: str) -> Dict:
    """
    Descarga y parsea las stats de un partido.
    Retorna dict listo para guardar en JSONL.
    """
    url = FEED_BASE.format(mid=match_id)
    req = urllib.request.Request(url, headers=FEED_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"fs_match_id": match_id, "scrape_status": "error", "error_msg": str(exc)}

    stats = parse_stats_feed(data)
    if stats is None:
        return {"fs_match_id": match_id, "scrape_status": "no_stats"}

    return {"fs_match_id": match_id, "scrape_status": "ok", **stats}


def load_checkpoint(jsonl_path: Path, skip_errors: bool = False) -> Set[str]:
    """
    Retorna el set de match_ids ya procesados en el archivo JSONL.
    Si skip_errors=True, solo considera registros con scrape_status='ok' como "done".
    """
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if skip_errors and rec.get("scrape_status") == "error":
                        continue
                    done.add(rec["fs_match_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def remove_errors_from_file(jsonl_path: Path) -> int:
    """Elimina registros con scrape_status='error' del JSONL. Retorna nº eliminados."""
    if not jsonl_path.exists():
        return 0
    keep = []
    removed = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if rec.get("scrape_status") == "error":
                        removed += 1
                    else:
                        keep.append(line)
                except json.JSONDecodeError:
                    keep.append(line)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in keep:
            f.write(line + "\n")
    return removed


def scrape_season_boxscores(
    season: str,
    delay: float = 1.5,
    resume: bool = False,
) -> Dict[str, int]:
    """
    Descarga estadísticas de todos los partidos de una temporada.
    Guarda incrementalmente en JSONL. Retorna resumen de conteos.
    """
    index_path = INDEX_DIR / f"{season}_index.json"
    if not index_path.exists():
        _print(f"[{season}] Índice no encontrado: {index_path}")
        return {}

    with open(index_path, encoding="utf-8") as f:
        matches = json.load(f)

    BOX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BOX_DIR / f"{season}_boxscores.jsonl"

    done: Set[str] = set()
    if resume:
        done = load_checkpoint(out_path, skip_errors=False)
        _print(f"[{season}] Checkpoint: {len(done)} ya procesados")

    pending = [m for m in matches if m["fs_match_id"] not in done]
    total = len(matches)
    _print(f"[{season}] {len(pending)} por procesar de {total} partidos")

    counts = {"ok": 0, "no_stats": 0, "error": 0}

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, match in enumerate(pending):
            mid = match["fs_match_id"]
            rec = fetch_match_stats(mid)

            # Añadir metadatos del índice
            rec["fs_date"]     = match.get("fs_date", "")
            rec["fs_home_raw"] = match.get("fs_home_raw", "")
            rec["fs_away_raw"] = match.get("fs_away_raw", "")
            rec["season"]      = season

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            status = rec["scrape_status"]
            counts[status] = counts.get(status, 0) + 1

            done_total = len(done) + i + 1
            if (i + 1) % 50 == 0 or (i + 1) == len(pending):
                pct_ok = counts["ok"] / max(1, sum(counts.values())) * 100
                _print(
                    f"  [{done_total}/{total}] {mid} → {status} | "
                    f"ok={counts['ok']} ({pct_ok:.0f}%) no_stats={counts['no_stats']} err={counts['error']}"
                )

            if i < len(pending) - 1:
                time.sleep(delay)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashScore BBL Boxscore Scraper")
    parser.add_argument("--season",  help="Una temporada (ej. 2024-25)")
    parser.add_argument("--resume",  action="store_true", help="Continúa desde checkpoint")
    parser.add_argument("--delay",        type=float, default=1.5, help="Segundos entre requests (default 1.5)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Antes de iniciar, elimina registros de error del JSONL para re-intentar")
    args = parser.parse_args()

    seasons = [args.season] if args.season else ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

    _print("=== FlashScore Boxscore Scraper BBL ===")
    _print(f"Temporadas: {seasons} | delay={args.delay}s | resume={args.resume} | retry-errors={args.retry_errors}")

    if args.retry_errors:
        for season in seasons:
            out_path = BOX_DIR / f"{season}_boxscores.jsonl"
            n_removed = remove_errors_from_file(out_path)
            if n_removed:
                _print(f"  [{season}] Eliminados {n_removed} registros de error")

    grand_total = {"ok": 0, "no_stats": 0, "error": 0}
    for season in seasons:
        counts = scrape_season_boxscores(season, delay=args.delay, resume=args.resume)
        for k, v in counts.items():
            grand_total[k] = grand_total.get(k, 0) + v
        total_s = sum(counts.values())
        if total_s:
            _print(
                f"[{season}] DONE: ok={counts.get('ok',0)} "
                f"no_stats={counts.get('no_stats',0)} "
                f"err={counts.get('error',0)} "
                f"({counts.get('ok',0)/total_s*100:.1f}% cobertura)\n"
            )

    _print("=== Resumen global ===")
    total_all = sum(grand_total.values())
    _print(
        f"  ok={grand_total['ok']} | "
        f"no_stats={grand_total['no_stats']} | "
        f"error={grand_total['error']} | "
        f"cobertura={grand_total['ok']/max(1,total_all)*100:.1f}%"
    )


if __name__ == "__main__":
    main()
