"""
Interfaz de predicción para la API.

Dado dos equipos (por nombre o club_id) y cuál juega en casa,
devuelve la probabilidad de victoria de cada uno.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "model", "artifacts", "model.pkl")
META_PATH     = os.path.join(BASE_DIR, "model", "artifacts", "model_meta.json")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MATCHES_PATH  = os.path.join(BASE_DIR, "data", "processed", "matches.csv")

_model = None
_meta  = None
_features_df: Optional[pd.DataFrame] = None
_matches_df:  Optional[pd.DataFrame] = None


def _load() -> None:
    global _model, _meta, _features_df, _matches_df
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _meta is None:
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    if _features_df is None:
        _features_df = pd.read_csv(FEATURES_PATH)
        _features_df["fecha"] = pd.to_datetime(_features_df["fecha"], utc=True)
    if _matches_df is None:
        _matches_df = pd.read_csv(MATCHES_PATH)
        _matches_df["fecha"] = pd.to_datetime(_matches_df["fecha"], utc=True)


def get_available_teams() -> List[Dict]:
    """
    Devuelve la lista de equipos disponibles (con su nombre canónico y club_id)
    ordenados por nombre.
    """
    _load()
    latest = (
        pd.concat([
            _matches_df[["club_local_id", "equipo_local", "fecha"]].rename(
                columns={"club_local_id": "club_id", "equipo_local": "nombre"}),
            _matches_df[["club_visitante_id", "equipo_visitante", "fecha"]].rename(
                columns={"club_visitante_id": "club_id", "equipo_visitante": "nombre"}),
        ])
        .sort_values("fecha")
        .groupby("club_id")["nombre"]
        .last()
        .reset_index()
        .sort_values("nombre")
    )
    return latest.to_dict(orient="records")


def _get_club_id(name_or_id) -> Optional[int]:
    """Resuelve nombre de equipo o ID a club_id."""
    _load()
    if isinstance(name_or_id, int):
        return name_or_id

    name_lower = str(name_or_id).lower().strip()
    local = _matches_df[["club_local_id", "equipo_local"]].rename(
        columns={"club_local_id": "club_id", "equipo_local": "nombre"})
    visit = _matches_df[["club_visitante_id", "equipo_visitante"]].rename(
        columns={"club_visitante_id": "club_id", "equipo_visitante": "nombre"})
    all_teams = pd.concat([local, visit]).drop_duplicates()

    # Búsqueda exacta primero
    exact = all_teams[all_teams["nombre"].str.lower() == name_lower]
    if not exact.empty:
        return int(exact.iloc[-1]["club_id"])

    # Búsqueda parcial
    partial = all_teams[all_teams["nombre"].str.lower().str.contains(name_lower, regex=False)]
    if not partial.empty:
        return int(partial.iloc[-1]["club_id"])

    return None


def _get_last_stats(club_id: int, n: int = 5) -> Dict[str, float]:
    """
    Calcula las estadísticas rolling de un equipo basadas en sus últimos N
    partidos en el dataset histórico.
    """
    _load()
    local_rows = _matches_df[_matches_df["club_local_id"] == club_id][[
        "fecha", "pts_local", "pts_visitante", "ganador",
        "loc_t2_pct", "loc_t3_pct", "loc_reb_totales", "loc_asistencias", "loc_valoracion"
    ]].rename(columns={
        "pts_local": "pts_for", "pts_visitante": "pts_against",
        "loc_t2_pct": "t2_pct", "loc_t3_pct": "t3_pct",
        "loc_reb_totales": "reb", "loc_asistencias": "ast", "loc_valoracion": "val",
    })
    local_rows["win"] = (local_rows["ganador"] == "local").astype(int)

    visit_rows = _matches_df[_matches_df["club_visitante_id"] == club_id][[
        "fecha", "pts_visitante", "pts_local", "ganador",
        "vis_t2_pct", "vis_t3_pct", "vis_reb_totales", "vis_asistencias", "vis_valoracion"
    ]].rename(columns={
        "pts_visitante": "pts_for", "pts_local": "pts_against",
        "vis_t2_pct": "t2_pct", "vis_t3_pct": "t3_pct",
        "vis_reb_totales": "reb", "vis_asistencias": "ast", "vis_valoracion": "val",
    })
    visit_rows["win"] = (visit_rows["ganador"] == "visitante").astype(int)

    history = pd.concat([local_rows, visit_rows]).sort_values("fecha").tail(n)

    if history.empty:
        return {}

    return {
        "pts_avg":           history["pts_for"].mean(),
        "pts_conceded_avg":  history["pts_against"].mean(),
        "win_rate":          history["win"].mean(),
        "t2_pct_avg":        history["t2_pct"].mean(),
        "t3_pct_avg":        history["t3_pct"].mean(),
        "reb_avg":           history["reb"].mean(),
        "ast_avg":           history["ast"].mean(),
        "val_avg":           history["val"].mean(),
        "streak":            history["win"].map({1: 1, 0: -1}).sum(),
    }


def _get_h2h(home_club_id: int, away_club_id: int, last_n: int = 10) -> Dict[str, float]:
    """Calcula estadísticas H2H entre dos clubes en el historial."""
    _load()
    mask = (
        ((_matches_df["club_local_id"] == home_club_id)
           & (_matches_df["club_visitante_id"] == away_club_id))
        | ((_matches_df["club_local_id"] == away_club_id)
           & (_matches_df["club_visitante_id"] == home_club_id))
    )
    h2h = _matches_df[mask].sort_values("fecha").tail(last_n)

    if h2h.empty:
        return {"h2h_home_wins": 0, "h2h_total": 0, "h2h_home_rate": 0.5}

    home_wins = int((
        ((_matches_df.loc[h2h.index, "club_local_id"] == home_club_id) & (h2h["ganador"] == "local")) |
        ((_matches_df.loc[h2h.index, "club_visitante_id"] == home_club_id) & (h2h["ganador"] == "visitante"))
    ).sum())

    total = len(h2h)
    return {
        "h2h_home_wins": int(home_wins),
        "h2h_total":     total,
        "h2h_home_rate": round(home_wins / total, 4),
    }


def predict(
    equipo_local: str | int,
    equipo_visitante: str | int,
) -> Dict:
    """
    Predice el resultado de un partido entre dos equipos.

    Args:
        equipo_local:     Nombre (parcial) o club_id del equipo local.
        equipo_visitante: Nombre (parcial) o club_id del equipo visitante.

    Returns:
        {
          "equipo_local":       str,
          "equipo_visitante":   str,
          "prob_local":         float,   # probabilidad de victoria del local
          "prob_visitante":     float,   # probabilidad de victoria del visitante
          "prediccion":         str,     # "local" | "visitante"
          "confianza":          str,     # "alta" | "media" | "baja"
          "features_usadas":    dict,    # features calculadas (para transparencia)
        }
    """
    _load()

    home_id = _get_club_id(equipo_local)
    away_id = _get_club_id(equipo_visitante)

    if home_id is None:
        raise ValueError(f"Equipo no encontrado: '{equipo_local}'")
    if away_id is None:
        raise ValueError(f"Equipo no encontrado: '{equipo_visitante}'")
    if home_id == away_id:
        raise ValueError("Los dos equipos deben ser distintos.")

    home_stats   = _get_last_stats(home_id, n=10)
    away_stats   = _get_last_stats(away_id, n=10)
    home_stats_3 = _get_last_stats(home_id, n=3)
    away_stats_3 = _get_last_stats(away_id, n=3)
    h2h          = _get_h2h(home_id, away_id)

    def s(stats: Dict, key: str, default: float = np.nan) -> float:
        return stats.get(key, default)

    features = {
        "home_pts_avg_5":           s(home_stats, "pts_avg"),
        "home_pts_conceded_avg_5":  s(home_stats, "pts_conceded_avg"),
        "home_win_rate_5":          s(home_stats, "win_rate"),
        "away_pts_avg_5":           s(away_stats, "pts_avg"),
        "away_pts_conceded_avg_5":  s(away_stats, "pts_conceded_avg"),
        "away_win_rate_5":          s(away_stats, "win_rate"),
        "pts_diff_5":               s(home_stats, "pts_avg") - s(away_stats, "pts_avg"),
        "win_rate_diff_5":          s(home_stats, "win_rate") - s(away_stats, "win_rate"),
        "home_pts_avg_10":          s(home_stats, "pts_avg"),
        "home_pts_conceded_avg_10": s(home_stats, "pts_conceded_avg"),
        "home_win_rate_10":         s(home_stats, "win_rate"),
        "away_pts_avg_10":          s(away_stats, "pts_avg"),
        "away_pts_conceded_avg_10": s(away_stats, "pts_conceded_avg"),
        "away_win_rate_10":         s(away_stats, "win_rate"),
        "pts_diff_10":              s(home_stats, "pts_avg") - s(away_stats, "pts_avg"),
        "win_rate_diff_10":         s(home_stats, "win_rate") - s(away_stats, "win_rate"),
        "home_t3_pct_avg_5":        s(home_stats, "t3_pct_avg"),
        "home_t2_pct_avg_5":        s(home_stats, "t2_pct_avg"),
        "home_reb_avg_5":           s(home_stats, "reb_avg"),
        "home_ast_avg_5":           s(home_stats, "ast_avg"),
        "home_val_avg_5":           s(home_stats, "val_avg"),
        "away_t3_pct_avg_5":        s(away_stats, "t3_pct_avg"),
        "away_t2_pct_avg_5":        s(away_stats, "t2_pct_avg"),
        "away_reb_avg_5":           s(away_stats, "reb_avg"),
        "away_ast_avg_5":           s(away_stats, "ast_avg"),
        "away_val_avg_5":           s(away_stats, "val_avg"),
        "val_diff_5":               s(home_stats, "val_avg") - s(away_stats, "val_avg"),
        "home_streak_5":            s(home_stats, "streak"),
        "away_streak_5":            s(away_stats, "streak"),
        "h2h_home_wins":            h2h["h2h_home_wins"],
        "h2h_total":                h2h["h2h_total"],
        "h2h_home_rate":            h2h["h2h_home_rate"],
        "is_home":                  1,
        "home_win_rate_3":          s(home_stats_3, "win_rate"),
        "away_win_rate_3":          s(away_stats_3, "win_rate"),
        "win_rate_diff_3":          s(home_stats_3, "win_rate") - s(away_stats_3, "win_rate"),
    }

    feature_cols = _meta["feature_cols"]
    X = pd.DataFrame([{c: features.get(c, np.nan) for c in feature_cols}])

    prob_local = float(_model.predict_proba(X)[0, 1])
    prob_visit = 1.0 - prob_local

    prediccion = "local" if prob_local >= 0.5 else "visitante"
    conf_val   = max(prob_local, prob_visit)
    confianza  = "alta" if conf_val >= 0.65 else ("media" if conf_val >= 0.55 else "baja")

    # Nombre canónico de equipos
    teams = {t["club_id"]: t["nombre"] for t in get_available_teams()}
    nombre_local = teams.get(home_id, str(equipo_local))
    nombre_visit = teams.get(away_id, str(equipo_visitante))

    return {
        "equipo_local":     nombre_local,
        "equipo_visitante": nombre_visit,
        "prob_local":       round(prob_local, 4),
        "prob_visitante":   round(prob_visit, 4),
        "prediccion":       prediccion,
        "confianza":        confianza,
        "features_usadas": {
            "home_win_rate_reciente":  float(round(s(home_stats, "win_rate"), 3)),
            "away_win_rate_reciente":  float(round(s(away_stats, "win_rate"), 3)),
            "home_pts_promedio":       float(round(s(home_stats, "pts_avg"), 1)),
            "away_pts_promedio":       float(round(s(away_stats, "pts_avg"), 1)),
            "home_val_promedio":       float(round(s(home_stats, "val_avg"), 1)),
            "away_val_promedio":       float(round(s(away_stats, "val_avg"), 1)),
            "h2h_enfrentamientos":     int(h2h["h2h_total"]),
            "h2h_tasa_local":          float(h2h["h2h_home_rate"]),
        },
    }
