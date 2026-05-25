"""
Fase 1: Índice de partidos BBL desde FlashScore.

Estrategia dual para cubrir el 100 % de los partidos:
  • Feed directo  (global.flashscore.ninja): temporada regular (Hauptrunde)
  • HTML de página de archivo (Playwright): playoffs + cierre de temporada regular

Salida: data/flashscore/index/{season}_index.json  (uno por temporada)

Uso:
    python -m scraper.flashscore_index_scraper                   # todas
    python -m scraper.flashscore_index_scraper --season 2024-25  # una temporada
    python -m scraper.flashscore_index_scraper --resume          # omite las ya hechas
    python -m scraper.flashscore_index_scraper --headful         # muestra el browser
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "flashscore" / "index"

# ── Configuración ──────────────────────────────────────────────────────────────
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# URL de la página de resultados de archivo por temporada
SEASON_ARCHIVE_URL = {
    "2020-21": "https://www.flashscore.com/basketball/germany/bbl-2020-2021/results/",
    "2021-22": "https://www.flashscore.com/basketball/germany/bbl-2021-2022/results/",
    "2022-23": "https://www.flashscore.com/basketball/germany/bbl-2022-2023/results/",
    "2023-24": "https://www.flashscore.com/basketball/germany/bbl-2023-2024/results/",
    "2024-25": "https://www.flashscore.com/basketball/germany/bbl-2024-2025/results/",
}

# ID numérico de la fase de temporada regular en FlashScore
# competition_id = ncAkL5qn (BBL), feed = tr_3_81_ncAkL5qn_{stage_id}_{page}_2_en_1
SEASON_STAGE_ID = {
    "2020-21": 171,
    "2021-22": 172,
    "2022-23": 176,
    "2023-24": 183,
    "2024-25": 184,
}

FEED_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.flashscore.com/",
    "Origin": "https://www.flashscore.com",
    "X-Fsign": "SW9D1eZo",
}
FEED_BASE = "https://global.flashscore.ninja/2/x/feed/tr_3_81_ncAkL5qn_{stage}_{page}_2_en_1"


# ── Parseo del formato SA÷ ─────────────────────────────────────────────────────
def parse_sa_data(data: str, season: str) -> Dict[str, Dict]:
    """
    Parsea el formato propietario FlashScore (SA÷/AA÷).
    Retorna dict {match_id: record} para deduplicación fácil.
    Cada registro va de ~AA÷... hasta el siguiente ~AA÷ (o fin).
    """
    records: Dict[str, Dict] = {}
    # Divide en registros individuales
    for raw in re.split(r'(?=~AA÷[A-Za-z0-9]{8})', data):
        if not raw.startswith("~AA÷"):
            continue
        mid_m = re.search(r"\bAA÷([A-Za-z0-9]{8})\b", raw)
        ts_m  = re.search(r"\bAD÷(\d{9,10})\b", raw)
        home_m = re.search(r"\bAE÷([^¬]+)", raw)
        away_m = re.search(r"\bAF÷([^¬]+)", raw)
        if not mid_m:
            continue
        mid = mid_m.group(1)
        try:
            fs_date = (
                datetime.fromtimestamp(int(ts_m.group(1)), tz=timezone.utc)
                .strftime("%Y-%m-%d")
                if ts_m
                else ""
            )
        except Exception:
            fs_date = ""
        # Team slug + ID  (WU÷=home_slug, PX÷=home_id, WV÷=away_slug, PY÷=away_id)
        home_slug_m = re.search(r"\bWU÷([^¬]+)", raw)
        home_id_m   = re.search(r"\bPX÷([A-Za-z0-9]+)", raw)
        away_slug_m = re.search(r"\bWV÷([^¬]+)", raw)
        away_id_m   = re.search(r"\bPY÷([A-Za-z0-9]+)", raw)

        records[mid] = {
            "fs_match_id": mid,
            "fs_date": fs_date,
            "fs_home_raw": home_m.group(1).strip() if home_m else "",
            "fs_away_raw": away_m.group(1).strip() if away_m else "",
            "fs_home_slug": home_slug_m.group(1).strip() if home_slug_m else "",
            "fs_home_id":   home_id_m.group(1).strip()   if home_id_m   else "",
            "fs_away_slug": away_slug_m.group(1).strip() if away_slug_m else "",
            "fs_away_id":   away_id_m.group(1).strip()   if away_id_m   else "",
            "season": season,
        }
    return records


# ── Parte 1: feed directo (temporada regular) ──────────────────────────────────
def fetch_regular_season(season: str) -> Dict[str, Dict]:
    """
    Descarga todas las páginas del feed de la temporada regular (Hauptrunde).
    No requiere navegador, usa urllib con las cabeceras de FlashScore.
    """
    stage_id = SEASON_STAGE_ID[season]
    all_records: Dict[str, Dict] = {}

    for page in range(1, 25):
        url = FEED_BASE.format(stage=stage_id, page=page)
        req = urllib.request.Request(url, headers=FEED_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"    Feed page {page}: error {exc}")
            break

        parsed = parse_sa_data(data, season)
        if not parsed:
            print(f"    Feed page {page}: vacío → fin de datos")
            break
        all_records.update(parsed)
        print(f"    Feed page {page}: {len(parsed)} partidos (acumulado: {len(all_records)})")

    return all_records


# ── Parte 2: página de archivo (playoffs + recientes) ─────────────────────────
def fetch_archive_page(season: str, headful: bool = False) -> Dict[str, Dict]:
    """
    Carga la página de resultados de archivo de FlashScore con Playwright.
    La primera carga incluye los partidos más recientes (playoffs + cierre
    de temporada regular), que NO están en el feed de la fase regular.
    """
    url = SEASON_ARCHIVE_URL[season]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not headful,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            user_agent=FEED_HEADERS["User-Agent"],
            locale="en-US",
            timezone_id="Europe/Berlin",
            viewport={"width": 1280, "height": 900},
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = ctx.new_page()
        page.goto(url, timeout=60_000, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)

        # Intenta cerrar cookie banner
        try:
            btn = page.locator("#onetrust-accept-btn-handler")
            if btn.count() > 0:
                btn.first.click(timeout=3000)
                page.wait_for_timeout(1000)
        except Exception:
            pass

        html = page.content()
        browser.close()

    return parse_sa_data(html, season)


# ── Combinar y guardar ─────────────────────────────────────────────────────────
def scrape_season_index(season: str, headful: bool = False) -> List[Dict]:
    print(f"\n[{season}] Iniciando…")

    # Parte 1: temporada regular vía feed directo
    print("  → Feed directo (temporada regular):")
    feed_records = fetch_regular_season(season)
    print(f"  Feed: {len(feed_records)} partidos")

    # Parte 2: página de archivo (playoffs + cierre)
    print("  → Playwright (playoffs + recientes):")
    html_records = fetch_archive_page(season, headful=headful)
    print(f"  HTML: {len(html_records)} partidos")

    # Merge (feed prevalece para datos de temporada regular, HTML añade los demás)
    combined = {**feed_records, **html_records}

    result = sorted(combined.values(), key=lambda r: r.get("fs_date", ""), reverse=True)
    dates = [r["fs_date"] for r in result if r["fs_date"]]
    date_range = f"{min(dates)}..{max(dates)}" if dates else "sin fechas"
    print(f"  TOTAL: {len(result)} partidos únicos | {date_range}")
    return result


def save_index(season: str, records: List[Dict]) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INDEX_DIR / f"{season}_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Guardado: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", help="Scrape una sola temporada (ej. 2024-25)")
    parser.add_argument("--resume", action="store_true",
                        help="Omite temporadas con archivo ya creado")
    parser.add_argument("--headful", action="store_true",
                        help="Muestra el browser (útil para depuración)")
    args = parser.parse_args()

    seasons_to_scrape = [args.season] if args.season else SEASONS

    if args.resume:
        pending = []
        for s in seasons_to_scrape:
            idx_file = INDEX_DIR / f"{s}_index.json"
            if idx_file.exists():
                with open(idx_file) as f:
                    existing = json.load(f)
                print(f"[{s}] Ya existe ({len(existing)} registros). Omitiendo.")
            else:
                pending.append(s)
        seasons_to_scrape = pending

    if not seasons_to_scrape:
        print("Nada que scrapear.")
        return

    print("=== FlashScore Index Scraper BBL ===")
    print(f"Temporadas: {seasons_to_scrape}")

    summary: Dict[str, int] = {}
    for i, season in enumerate(seasons_to_scrape):
        records = scrape_season_index(season, headful=args.headful)
        save_index(season, records)
        summary[season] = len(records)
        if i < len(seasons_to_scrape) - 1:
            print("  Pausa 5s…")
            time.sleep(5)

    print("\n=== Resumen ===")
    total = 0
    for season, count in summary.items():
        print(f"  {season}: {count} partidos")
        total += count
    print(f"  TOTAL: {total} partidos en {len(summary)} temporadas")


if __name__ == "__main__":
    main()
