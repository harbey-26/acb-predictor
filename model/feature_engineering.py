"""
Fase 2: Feature Engineering para predicción de partidos ACB.

Uso:
    python -m model.feature_engineering

Genera data/processed/features.csv con todas las features calculadas
sin data leakage (solo usa información de partidos anteriores).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw",
)
PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed",
)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "features.csv")

WINDOWS = [5, 10]
H2H_WINDOW = 10


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------

def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Lee todos los JSON de raw_dir y devuelve un DataFrame."""
    records: List[dict] = []
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            records.append(json.load(fh))

    if not records:
        raise FileNotFoundError(f"No se encontraron JSONs en {raw_dir}")

    df = pd.DataFrame(records)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df = df.sort_values("fecha", kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Normalización de nombres de equipo
# ---------------------------------------------------------------------------

def build_club_id_map(df: pd.DataFrame) -> Dict[int, str]:
    """
    Construye un mapeo club_id -> nombre canónico.
    Usa el nombre más reciente (el que aparece en el partido con fecha mayor).
    """
    # Apilamos apariciones local y visitante
    local_pairs = df[["club_local_id", "equipo_local", "fecha"]].rename(
        columns={"club_local_id": "club_id", "equipo_local": "nombre"}
    )
    visit_pairs = df[["club_visitante_id", "equipo_visitante", "fecha"]].rename(
        columns={"club_visitante_id": "club_id", "equipo_visitante": "nombre"}
    )
    pairs = pd.concat([local_pairs, visit_pairs], ignore_index=True)
    # Nombre más reciente por club_id
    latest = pairs.sort_values("fecha").groupby("club_id")["nombre"].last()
    return latest.to_dict()


# ---------------------------------------------------------------------------
# 3. Construcción del historial por equipo
# ---------------------------------------------------------------------------

def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame largo con una fila por (partido, equipo),
    con columnas normalizadas: club_id, pts_for, pts_against, win, t2_pct,
    t3_pct, reb_totales, asistencias, valoracion, fecha, match_id.
    """
    rows_local = df[[
        "match_id", "fecha", "club_local_id",
        "pts_local", "pts_visitante", "ganador",
        "loc_t2_pct", "loc_t3_pct",
        "loc_reb_totales", "loc_asistencias", "loc_valoracion",
    ]].copy()
    rows_local = rows_local.rename(columns={
        "club_local_id": "club_id",
        "pts_local": "pts_for",
        "pts_visitante": "pts_against",
        "loc_t2_pct": "t2_pct",
        "loc_t3_pct": "t3_pct",
        "loc_reb_totales": "reb_totales",
        "loc_asistencias": "asistencias",
        "loc_valoracion": "valoracion",
    })
    rows_local["win"] = (rows_local["ganador"] == "local").astype(int)

    rows_visit = df[[
        "match_id", "fecha", "club_visitante_id",
        "pts_visitante", "pts_local", "ganador",
        "vis_t2_pct", "vis_t3_pct",
        "vis_reb_totales", "vis_asistencias", "vis_valoracion",
    ]].copy()
    rows_visit = rows_visit.rename(columns={
        "club_visitante_id": "club_id",
        "pts_visitante": "pts_for",
        "pts_local": "pts_against",
        "vis_t2_pct": "t2_pct",
        "vis_t3_pct": "t3_pct",
        "vis_reb_totales": "reb_totales",
        "vis_asistencias": "asistencias",
        "vis_valoracion": "valoracion",
    })
    rows_visit["win"] = (rows_visit["ganador"] == "visitante").astype(int)

    history = pd.concat([rows_local, rows_visit], ignore_index=True)
    history = history.drop(columns=["ganador"])
    history = history.sort_values(["club_id", "fecha", "match_id"], kind="mergesort")
    history = history.reset_index(drop=True)
    return history


# ---------------------------------------------------------------------------
# 4. Rolling stats por equipo (sin data leakage)
# ---------------------------------------------------------------------------

def compute_rolling_stats(
    history: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Para cada fila en history (un partido de un equipo), calcula las
    estadísticas rolling de los últimos `window` partidos ANTERIORES.
    Devuelve un DataFrame con match_id, club_id y las columnas rolling.
    """
    stat_cols = ["pts_for", "pts_against", "win", "t2_pct", "t3_pct",
                 "reb_totales", "asistencias", "valoracion"]

    results: List[pd.DataFrame] = []

    for club_id, grp in history.groupby("club_id", sort=False):
        grp = grp.sort_values("fecha", kind="mergesort").reset_index(drop=True)

        rolled: Dict[str, pd.Series] = {}
        for col in stat_cols:
            # shift(1) garantiza que el partido actual NO se incluye
            rolled[f"{col}_avg_{window}"] = (
                grp[col]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

        # Racha: win=1 -> +1, loss=0 -> -1, sumamos en ventana
        streak_raw = grp["win"].map({1: 1, 0: -1})
        rolled[f"streak_{window}"] = (
            streak_raw
            .shift(1)
            .rolling(window=window, min_periods=1)
            .sum()
        )

        rolled_df = pd.DataFrame(rolled)
        rolled_df["match_id"] = grp["match_id"].values
        rolled_df["club_id"] = club_id
        results.append(rolled_df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Head-to-head
# ---------------------------------------------------------------------------

def compute_h2h(
    df_lr: pd.DataFrame,
    window: int = H2H_WINDOW,
) -> pd.DataFrame:
    """
    Para cada partido, calcula las features H2H entre los dos clubes
    usando solo partidos ANTERIORES (fecha < fecha_partido).

    Devuelve un DataFrame con match_id, h2h_home_wins, h2h_total, h2h_home_rate.
    """
    # Construimos lookup: (club_a, club_b) -> lista de (fecha, ganó_club_a)
    # donde club_a < club_b para normalizar la pareja

    # Ordenamos por fecha para iterar cronológicamente
    df_sorted = df_lr.sort_values("fecha", kind="mergesort").reset_index(drop=True)

    # Historial acumulado por pareja canónica
    # Clave: (min_club, max_club) -> list of (fecha, winner_club_id)
    pair_history: Dict[Tuple[int, int], List[Tuple[pd.Timestamp, int]]] = {}

    h2h_home_wins_list: List[int] = []
    h2h_total_list: List[int] = []
    match_id_list: List[int] = []

    for _, row in df_sorted.iterrows():
        home_id = int(row["club_local_id"])
        away_id = int(row["club_visitante_id"])
        key = (min(home_id, away_id), max(home_id, away_id))
        fecha = row["fecha"]
        winner_id = home_id if row["ganador"] == "local" else away_id

        # Calcular H2H con historial previo
        hist = pair_history.get(key, [])
        # Solo los últimos `window` enfrentamientos antes de este partido
        relevant = hist[-window:]  # ya están ordenados por fecha (insertamos en orden)

        total = len(relevant)
        home_wins = sum(1 for (_, w) in relevant if w == home_id)

        h2h_home_wins_list.append(home_wins)
        h2h_total_list.append(total)
        match_id_list.append(int(row["match_id"]))

        # Añadir este partido al historial
        if key not in pair_history:
            pair_history[key] = []
        pair_history[key].append((fecha, winner_id))

    result = pd.DataFrame({
        "match_id": match_id_list,
        "h2h_home_wins": h2h_home_wins_list,
        "h2h_total": h2h_total_list,
    })
    result["h2h_home_rate"] = result.apply(
        lambda r: r["h2h_home_wins"] / r["h2h_total"] if r["h2h_total"] > 0 else 0.5,
        axis=1,
    )
    return result


# ---------------------------------------------------------------------------
# 6. Construcción del dataset final
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame completo (todos los round_types).
    Devuelve un DataFrame de features para partidos de liga regular (LR).
    """
    # Filtrar liga regular para features y target
    df_lr = df[df["round_type"] == "LR"].copy().reset_index(drop=True)

    if df_lr.empty:
        raise ValueError("No hay partidos de liga regular (round_type=='LR') en los datos.")

    # Historial con TODOS los partidos (LR únicamente, para no contaminar)
    # Usamos solo LR también para el historial por equipo, coherente con el filtro
    history = build_team_history(df_lr)

    # Calcular rolling stats para cada ventana
    rolling_dfs: Dict[int, pd.DataFrame] = {}
    for w in WINDOWS:
        rolling_dfs[w] = compute_rolling_stats(history, window=w)

    # Construir lookup: (match_id, club_id) -> stats
    def make_lookup(rolled_df: pd.DataFrame) -> Dict[Tuple[int, int], Dict[str, float]]:
        lookup: Dict[Tuple[int, int], Dict[str, float]] = {}
        for _, row in rolled_df.iterrows():
            key = (int(row["match_id"]), int(row["club_id"]))
            lookup[key] = row.drop(["match_id", "club_id"]).to_dict()
        return lookup

    lookups = {w: make_lookup(rolling_dfs[w]) for w in WINDOWS}

    # H2H
    h2h_df = compute_h2h(df_lr)
    h2h_lookup: Dict[int, Dict[str, float]] = {}
    for _, row in h2h_df.iterrows():
        h2h_lookup[int(row["match_id"])] = {
            "h2h_home_wins": row["h2h_home_wins"],
            "h2h_total": row["h2h_total"],
            "h2h_home_rate": row["h2h_home_rate"],
        }

    # Ensamblar features
    feature_rows: List[Dict] = []

    for _, match in df_lr.iterrows():
        mid = int(match["match_id"])
        home_id = int(match["club_local_id"])
        away_id = int(match["club_visitante_id"])

        row: Dict = {
            "match_id": mid,
            "temporada": match["temporada"],
            "fecha": match["fecha"],
            "jornada": match["jornada"],
            "equipo_local": match["equipo_local"],
            "equipo_visitante": match["equipo_visitante"],
            "club_local_id": home_id,
            "club_visitante_id": away_id,
        }

        # Rolling stats por ventana
        for w in WINDOWS:
            home_stats = lookups[w].get((mid, home_id), {})
            away_stats = lookups[w].get((mid, away_id), {})

            row[f"home_pts_avg_{w}"] = home_stats.get(f"pts_for_avg_{w}", np.nan)
            row[f"home_pts_conceded_avg_{w}"] = home_stats.get(f"pts_against_avg_{w}", np.nan)
            row[f"home_win_rate_{w}"] = home_stats.get(f"win_avg_{w}", np.nan)

            row[f"away_pts_avg_{w}"] = away_stats.get(f"pts_for_avg_{w}", np.nan)
            row[f"away_pts_conceded_avg_{w}"] = away_stats.get(f"pts_against_avg_{w}", np.nan)
            row[f"away_win_rate_{w}"] = away_stats.get(f"win_avg_{w}", np.nan)

            row[f"pts_diff_{w}"] = (
                row[f"home_pts_avg_{w}"] - row[f"away_pts_avg_{w}"]
                if not (np.isnan(row[f"home_pts_avg_{w}"]) or np.isnan(row[f"away_pts_avg_{w}"]))
                else np.nan
            )
            row[f"win_rate_diff_{w}"] = (
                row[f"home_win_rate_{w}"] - row[f"away_win_rate_{w}"]
                if not (np.isnan(row[f"home_win_rate_{w}"]) or np.isnan(row[f"away_win_rate_{w}"]))
                else np.nan
            )

        # Features adicionales solo con ventana 5
        w5 = 5
        home_stats5 = lookups[w5].get((mid, home_id), {})
        away_stats5 = lookups[w5].get((mid, away_id), {})

        row["home_t3_pct_avg_5"] = home_stats5.get(f"t3_pct_avg_{w5}", np.nan)
        row["home_t2_pct_avg_5"] = home_stats5.get(f"t2_pct_avg_{w5}", np.nan)
        row["home_reb_avg_5"] = home_stats5.get(f"reb_totales_avg_{w5}", np.nan)
        row["home_ast_avg_5"] = home_stats5.get(f"asistencias_avg_{w5}", np.nan)
        row["home_val_avg_5"] = home_stats5.get(f"valoracion_avg_{w5}", np.nan)

        row["away_t3_pct_avg_5"] = away_stats5.get(f"t3_pct_avg_{w5}", np.nan)
        row["away_t2_pct_avg_5"] = away_stats5.get(f"t2_pct_avg_{w5}", np.nan)
        row["away_reb_avg_5"] = away_stats5.get(f"reb_totales_avg_{w5}", np.nan)
        row["away_ast_avg_5"] = away_stats5.get(f"asistencias_avg_{w5}", np.nan)
        row["away_val_avg_5"] = away_stats5.get(f"valoracion_avg_{w5}", np.nan)

        val_diff = (
            row["home_val_avg_5"] - row["away_val_avg_5"]
            if not (np.isnan(row["home_val_avg_5"]) or np.isnan(row["away_val_avg_5"]))
            else np.nan
        )
        row["val_diff_5"] = val_diff

        # Racha
        row["home_streak_5"] = home_stats5.get(f"streak_{w5}", np.nan)
        row["away_streak_5"] = away_stats5.get(f"streak_{w5}", np.nan)

        # H2H
        h2h = h2h_lookup.get(mid, {})
        row["h2h_home_wins"] = h2h.get("h2h_home_wins", 0)
        row["h2h_total"] = h2h.get("h2h_total", 0)
        row["h2h_home_rate"] = h2h.get("h2h_home_rate", 0.5)

        # Factor local (siempre 1, captura la ventaja de campo)
        row["is_home"] = 1

        # Target
        row["target"] = 1 if match["ganador"] == "local" else 0

        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)
    return features_df


# ---------------------------------------------------------------------------
# 7. Guardado y resumen
# ---------------------------------------------------------------------------

def save_and_report(features_df: pd.DataFrame, output_path: str) -> None:
    """Guarda el CSV y muestra un resumen."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)

    n_rows, n_cols = features_df.shape
    target_rate = features_df["target"].mean() * 100

    print("=" * 60)
    print("RESUMEN DEL DATASET DE FEATURES")
    print("=" * 60)
    print(f"Filas      : {n_rows}")
    print(f"Columnas   : {n_cols}")
    print(f"Target=1   : {target_rate:.1f}% (victorias del local)")
    print()

    # Columnas de features (excluir metadatos y target)
    meta_cols = {
        "match_id", "temporada", "fecha", "jornada",
        "equipo_local", "equipo_visitante",
        "club_local_id", "club_visitante_id",
        "target",
    }
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    print(f"Features ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  {col}")
    print()

    # Nulos por columna de feature
    null_pct = features_df[feature_cols].isnull().mean() * 100
    cols_with_nulls = null_pct[null_pct > 0].sort_values(ascending=False)
    if cols_with_nulls.empty:
        print("No hay valores nulos en las features.")
    else:
        print("Valores nulos por columna:")
        for col, pct in cols_with_nulls.items():
            print(f"  {col}: {pct:.1f}%")

    print()
    print(f"Archivo guardado en: {output_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Cargando datos crudos...")
    df = load_raw_data(RAW_DIR)
    print(f"  {len(df)} partidos cargados (todos los round_types).")

    lr_count = (df["round_type"] == "LR").sum()
    print(f"  {lr_count} partidos de Liga Regular (LR).")

    print("Construyendo features...")
    features_df = build_features(df)

    print("Guardando y generando resumen...")
    save_and_report(features_df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
