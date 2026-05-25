"""
Fase 3: Fusión de boxscores FlashScore con bbl_matches.csv.

Une las estadísticas scrapeadas con el CSV de partidos usando
fecha + equipo_local + equipo_visitante como clave de matching.

Estrategia:
  1. Normalizar nombres de equipos (FS abreviado → CSV completo)
  2. Match exacto por (fecha, home, away)
  3. Log de no-matches para revisión manual

Salida:
  data/processed/bbl_matches_enriched.csv
  data/flashscore/unmatched_log.csv

Uso:
    python -m scraper.bbl_merge_boxscores
    python -m scraper.bbl_merge_boxscores --check-only   # solo estadísticas, no guarda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
MATCHES_CSV    = BASE_DIR / "data" / "processed" / "bbl_matches.csv"
ENRICHED_CSV   = BASE_DIR / "data" / "processed" / "bbl_matches_enriched.csv"
INDEX_DIR      = BASE_DIR / "data" / "flashscore" / "index"
BOXSCORES_DIR  = BASE_DIR / "data" / "flashscore" / "boxscores"
UNMATCHED_LOG  = BASE_DIR / "data" / "flashscore" / "unmatched_log.csv"

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# Normalización: FS raw name (lowercase) → nombre canónico del CSV
FS_TO_CSV: Dict[str, str] = {
    "alba berlin":           "ALBA Berlin",
    "alba":                  "ALBA Berlin",
    "bayern":                "FC Bayern Muenchen Basketball",
    "fc bayern":             "FC Bayern Muenchen Basketball",
    "bamberg":               "Bamberg Baskets",
    "basketball braunschweig": "Basketball Loewen Braunschweig",
    "braunschweig":          "Basketball Loewen Braunschweig",
    "bayreuth":              "Medi Bayreuth",
    "medi bayreuth":         "Medi Bayreuth",
    "bonn":                  "Telekom Baskets Bonn",
    "telekom baskets bonn":  "Telekom Baskets Bonn",
    "chemnitz":              "NINERS Chemnitz",
    "niners chemnitz":       "NINERS Chemnitz",
    "crailsheim merlins":    "Merlins Crailsheim",
    "merlins crailsheim":    "Merlins Crailsheim",
    "crailsheim":            "Merlins Crailsheim",
    "merlins":               "Merlins Crailsheim",
    "frankfurt":             "Fraport Skyliners Frankfurt",
    "skyliners frankfurt":   "Fraport Skyliners Frankfurt",
    "fraport skyliners":     "Fraport Skyliners Frankfurt",
    "giessen":               "46ers Giessen",
    "46ers giessen":         "46ers Giessen",
    "giessen 46ers":         "46ers Giessen",
    "gottingen":             "BG Goettingen",
    "goettingen":            "BG Goettingen",
    "bg goettingen":         "BG Goettingen",
    "hamburg":               "Veolia Towers Hamburg",
    "veolia towers hamburg": "Veolia Towers Hamburg",
    "towers hamburg":        "Veolia Towers Hamburg",
    "heidelberg":            "MLP Academics Heidelberg",
    "mlp academics":         "MLP Academics Heidelberg",
    "academics heidelberg":  "MLP Academics Heidelberg",
    "ludwigsburg":           "MHP Riesen Ludwigsburg",
    "mhp riesen":            "MHP Riesen Ludwigsburg",
    "riesen ludwigsburg":    "MHP Riesen Ludwigsburg",
    "oldenburg":             "EWE Baskets Oldenburg",
    "ewe baskets":           "EWE Baskets Oldenburg",
    "ewe baskets oldenburg": "EWE Baskets Oldenburg",
    "rostock":               "Rostock Seawolves",
    "rostock seawolves":     "Rostock Seawolves",
    "syntainics mbc":        "SYNTAINICS MBC Weissenfels",
    "mbc weissenfels":       "SYNTAINICS MBC Weissenfels",
    "syntainics":            "SYNTAINICS MBC Weissenfels",
    "mbc":                   "SYNTAINICS MBC Weissenfels",
    "ulm":                   "Ratiopharm Ulm",
    "ratiopharm ulm":        "Ratiopharm Ulm",
    "ratiopharm":            "Ratiopharm Ulm",
    "vechta":                "Rasta Vechta",
    "rasta vechta":          "Rasta Vechta",
    "wurzburg":              "FITOne Wuerzburg Baskets",
    "wuerzburg":             "FITOne Wuerzburg Baskets",
    "fitone wuerzburg":      "FITOne Wuerzburg Baskets",
    "paderborn":             "Baskets Paderborn",
    "baskets paderborn":     "Baskets Paderborn",
    # Tiger Tübingen (neoPhoenix) — not in current bbl_matches.csv
    "tubingen":              "neoPhoenix Tubingen",
    "neophoenix":            "neoPhoenix Tubingen",
    "tigers tubingen":       "neoPhoenix Tubingen",
}

# Columnas de boxscore que se añadirán al CSV
BOXSCORE_COLS = [
    "home_fga", "home_fgm", "home_fg_pct",
    "home_fg2a", "home_fg2m", "home_fg2_pct",
    "home_fg3a", "home_fg3m", "home_fg3_pct",
    "home_fta",  "home_ftm",  "home_ft_pct",
    "home_oreb", "home_dreb", "home_reb",
    "home_ast",  "home_blk",  "home_tov",  "home_stl",  "home_pf",
    "away_fga", "away_fgm", "away_fg_pct",
    "away_fg2a", "away_fg2m", "away_fg2_pct",
    "away_fg3a", "away_fg3m", "away_fg3_pct",
    "away_fta",  "away_ftm",  "away_ft_pct",
    "away_oreb", "away_dreb", "away_reb",
    "away_ast",  "away_blk",  "away_tov",  "away_stl",  "away_pf",
]


def normalize_team(name: str) -> str:
    """
    Convierte un nombre de equipo (CSV o FlashScore) al nombre canónico del CSV.
    """
    key = name.lower().strip()
    if key in FS_TO_CSV:
        return FS_TO_CSV[key]
    # Búsqueda por subclave (palabras)
    for k, v in FS_TO_CSV.items():
        if k in key or key in k:
            return v
    # Si ya está en formato CSV (nombre largo), lo devolvemos tal cual
    return name


def load_boxscores() -> pd.DataFrame:
    """
    Carga todos los JSONL de boxscores en un único DataFrame.
    Incluye solo los registros con scrape_status='ok'.
    """
    records: List[Dict] = []
    for season in SEASONS:
        path = BOXSCORES_DIR / f"{season}_boxscores.jsonl"
        if not path.exists():
            print(f"  [WARN] No encontrado: {path}")
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("scrape_status") == "ok":
                        records.append(rec)
                        count += 1
                except json.JSONDecodeError:
                    pass
        print(f"  {season}: {count} boxscores con stats")
    return pd.DataFrame(records) if records else pd.DataFrame()


def load_index() -> pd.DataFrame:
    """Carga todos los índices en un único DataFrame con info de equipo y fecha."""
    rows: List[Dict] = []
    for season in SEASONS:
        path = INDEX_DIR / f"{season}_index.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                rows.append({
                    "fs_match_id": r["fs_match_id"],
                    "fs_date":     r["fs_date"],
                    "fs_home_raw": r["fs_home_raw"],
                    "fs_away_raw": r["fs_away_raw"],
                    "season":      r["season"],
                })
    return pd.DataFrame(rows)


def _swap_home_away_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una copia del DataFrame con las columnas home_* y away_* intercambiadas.
    Se usa cuando FS tiene home/away invertido respecto al CSV (local/visitante).
    """
    df_swapped = df.copy()
    home_cols = [c for c in BOXSCORE_COLS if c.startswith("home_")]
    away_cols = [c for c in BOXSCORE_COLS if c.startswith("away_")]
    for hc, ac in zip(home_cols, away_cols):
        df_swapped[hc], df_swapped[ac] = df[ac].copy(), df[hc].copy()
    return df_swapped


def merge(check_only: bool = False) -> None:
    print("=== BBL Merge Boxscores ===")

    # 1. Cargar CSV de partidos
    df = pd.read_csv(MATCHES_CSV)
    df["fecha_dt"] = pd.to_datetime(df["fecha"], utc=True).dt.strftime("%Y-%m-%d")
    print(f"Partidos CSV: {len(df)}")

    # 2. Cargar índice FlashScore
    idx_df = load_index()
    print(f"Índice FlashScore: {len(idx_df)} partidos")

    # 3. Cargar boxscores
    print("Cargando boxscores…")
    box_df = load_boxscores()
    if box_df.empty:
        print("[ERROR] No hay boxscores. Ejecuta flashscore_boxscore_scraper.py primero.")
        sys.exit(1)
    print(f"Total boxscores ok: {len(box_df)}")

    # 4. Unir índice + boxscores (por fs_match_id)
    merged_box = idx_df.merge(box_df[["fs_match_id"] + BOXSCORE_COLS], on="fs_match_id", how="inner")
    print(f"Índice+boxscores unidos: {len(merged_box)}")

    # 5. Normalizar nombres de equipos en FS
    merged_box["home_norm"] = merged_box["fs_home_raw"].apply(normalize_team)
    merged_box["away_norm"] = merged_box["fs_away_raw"].apply(normalize_team)

    # 6. Crear claves de match en CSV (usando nombres originales del CSV)
    df["match_key"] = df["fecha_dt"] + "|" + df["equipo_local"] + "|" + df["equipo_visitante"]

    # 7a. Clave forward: FS home = CSV local (sin swap)
    merged_box["match_key_fwd"] = (
        merged_box["fs_date"] + "|" + merged_box["home_norm"] + "|" + merged_box["away_norm"]
    )

    # 7b. Clave reverse: FS home = CSV visitante (con swap de home/away)
    merged_box["match_key_rev"] = (
        merged_box["fs_date"] + "|" + merged_box["away_norm"] + "|" + merged_box["home_norm"]
    )

    # 8a. Merge forward (sin swap)
    fwd_cols = ["match_key_fwd", "fs_match_id"] + BOXSCORE_COLS
    enriched = df.merge(
        merged_box[fwd_cols].rename(columns={"match_key_fwd": "match_key"}),
        on="match_key",
        how="left",
        suffixes=("", "_fwd"),
    )
    enriched["fs_match_orientation"] = enriched["fs_match_id"].apply(
        lambda x: "forward" if pd.notna(x) else None
    )

    # 8b. Merge reverse para los no-matched (swap home/away stats)
    unmatched_mask = enriched["fs_match_id"].isna()
    n_unmatched = unmatched_mask.sum()
    print(f"  Forward matched: {len(df) - n_unmatched}/{len(df)}")

    if n_unmatched > 0:
        # Crear versión swapped de merged_box con clave reverse
        merged_box_swapped = _swap_home_away_cols(merged_box)
        rev_cols = ["match_key_rev", "fs_match_id"] + BOXSCORE_COLS
        df_unmatched = df[unmatched_mask][["match_key", "game_id"]].copy()
        rev_merged = df_unmatched.merge(
            merged_box_swapped[rev_cols].rename(columns={"match_key_rev": "match_key"}),
            on="match_key",
            how="left",
        ).set_index("game_id")

        # Aplicar stats reverse a los registros unmatched
        for col in ["fs_match_id"] + BOXSCORE_COLS:
            if col in rev_merged.columns:
                enriched.loc[unmatched_mask, col] = (
                    enriched.loc[unmatched_mask, "game_id"].map(rev_merged[col])
                )
        enriched.loc[unmatched_mask, "fs_match_orientation"] = (
            enriched.loc[unmatched_mask, "fs_match_id"].apply(
                lambda x: "reverse" if pd.notna(x) else None
            )
        )
        n_rev_matched = enriched[unmatched_mask]["fs_match_id"].notna().sum()
        print(f"  Reverse matched: {n_rev_matched}/{n_unmatched}")

    # 9. Estadísticas
    matched = enriched["fs_match_id"].notna().sum()
    total = len(enriched)
    unmatched = total - matched
    print(f"\nMatched: {matched}/{total} ({matched/total*100:.1f}%)")
    print(f"Sin match: {unmatched} ({unmatched/total*100:.1f}%)")

    # 10. Log de no-matches
    unmatched_df = enriched[enriched["fs_match_id"].isna()][
        ["game_id", "fecha_dt", "equipo_local", "equipo_visitante", "temporada"]
    ]
    if not unmatched_df.empty:
        print("\n--- Partidos sin match (primeros 20) ---")
        print(unmatched_df.head(20).to_string(index=False))

        if not check_only:
            UNMATCHED_LOG.parent.mkdir(parents=True, exist_ok=True)
            unmatched_df.to_csv(UNMATCHED_LOG, index=False)
            print(f"\nLog guardado: {UNMATCHED_LOG}")

    # 11. Desglose por temporada
    print("\nCobertura por temporada:")
    for season in SEASONS:
        mask = enriched["temporada"] == season
        n_total = mask.sum()
        n_ok = enriched[mask]["fs_match_id"].notna().sum()
        pct = n_ok / n_total * 100 if n_total else 0
        print(f"  {season}: {n_ok}/{n_total} ({pct:.1f}%)")

    if check_only:
        print("\n[check-only] No se guardó ningún archivo.")
        return

    # 12. Limpiar columnas auxiliares y guardar
    cols_to_drop = ["fecha_dt", "match_key"]
    enriched = enriched.drop(columns=[c for c in cols_to_drop if c in enriched.columns])
    enriched.to_csv(ENRICHED_CSV, index=False)
    print(f"\nGuardado: {ENRICHED_CSV} ({len(enriched)} filas, {len(enriched.columns)} columnas)")
    print(f"Columnas nuevas: {BOXSCORE_COLS[:5]}…  (+{len(BOXSCORE_COLS)-5} más)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="Solo muestra estadísticas de match, no guarda archivos")
    args = parser.parse_args()
    merge(check_only=args.check_only)


if __name__ == "__main__":
    main()
