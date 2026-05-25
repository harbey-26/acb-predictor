"""
Interfaz de predicción BBL para la API.
Carga bbl_model.pkl y bbl_matches.csv para predicciones en tiempo real.
"""

from __future__ import annotations

import json
import os
from datetime import date, timezone
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "artifacts", "bbl_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model", "artifacts", "bbl_model_meta.json")
MATCHES_PATH = os.path.join(BASE_DIR, "data", "processed", "bbl_matches.csv")

ELO_INITIAL = 1500
ELO_K = 20
ELO_HOME_ADVANTAGE = 100
ELO_SEASON_REGRESSION = 1 / 3
MAX_DAYS_REST = 30

_model = None
_meta = None
_matches_df: Optional[pd.DataFrame] = None
_elo_ratings: Optional[Dict[int, float]] = None   # club_id → ELO actual
_last_game_date: Optional[Dict[int, pd.Timestamp]] = None  # club_id → fecha último partido


def _compute_elo_and_rest() -> None:
    """Recalcula ELO y última fecha de partido para todos los equipos desde bbl_matches.csv."""
    global _elo_ratings, _last_game_date
    ratings: Dict[int, float] = {}
    last_date: Dict[int, pd.Timestamp] = {}
    prev_season: Optional[str] = None

    for _, row in _matches_df.sort_values("fecha", kind="mergesort").iterrows():
        home_id = int(row["club_local_id"])
        away_id = int(row["club_visitante_id"])
        season = row["temporada"]

        if season != prev_season and prev_season is not None:
            for tid in list(ratings.keys()):
                ratings[tid] = ratings[tid] * (1 - ELO_SEASON_REGRESSION) + ELO_INITIAL * ELO_SEASON_REGRESSION
        prev_season = season

        if home_id not in ratings:
            ratings[home_id] = ELO_INITIAL
        if away_id not in ratings:
            ratings[away_id] = ELO_INITIAL

        home_elo = ratings[home_id]
        away_elo = ratings[away_id]

        home_elo_adj = home_elo + ELO_HOME_ADVANTAGE
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo_adj) / 400))
        actual_home = 1.0 if row["ganador"] == "local" else 0.0

        ratings[home_id] = home_elo + ELO_K * (actual_home - exp_home)
        ratings[away_id] = away_elo + ELO_K * ((1 - actual_home) - (1 - exp_home))
        last_date[home_id] = row["fecha"]
        last_date[away_id] = row["fecha"]

    _elo_ratings = ratings
    _last_game_date = last_date


def _load() -> None:
    global _model, _meta, _matches_df
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _meta is None:
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    if _matches_df is None:
        _matches_df = pd.read_csv(MATCHES_PATH)
        _matches_df["fecha"] = pd.to_datetime(_matches_df["fecha"], utc=True)
        _compute_elo_and_rest()


def is_available() -> bool:
    """True si el modelo BBL existe (scrapeado y entrenado)."""
    return os.path.exists(MODEL_PATH) and os.path.exists(MATCHES_PATH)


def get_available_teams() -> List[Dict]:
    _load()
    local = _matches_df[["club_local_id", "equipo_local", "fecha"]].rename(
        columns={"club_local_id": "club_id", "equipo_local": "nombre"})
    visit = _matches_df[["club_visitante_id", "equipo_visitante", "fecha"]].rename(
        columns={"club_visitante_id": "club_id", "equipo_visitante": "nombre"})
    latest = (
        pd.concat([local, visit])
        .sort_values("fecha")
        .groupby("club_id")["nombre"]
        .last()
        .reset_index()
        .sort_values("nombre")
    )
    return latest.to_dict(orient="records")


def _get_club_id(name_or_id) -> Optional[int]:
    _load()
    if isinstance(name_or_id, int):
        return name_or_id
    name_lower = str(name_or_id).lower().strip()
    local = _matches_df[["club_local_id", "equipo_local"]].rename(
        columns={"club_local_id": "club_id", "equipo_local": "nombre"})
    visit = _matches_df[["club_visitante_id", "equipo_visitante"]].rename(
        columns={"club_visitante_id": "club_id", "equipo_visitante": "nombre"})
    all_teams = pd.concat([local, visit]).drop_duplicates()
    exact = all_teams[all_teams["nombre"].str.lower() == name_lower]
    if not exact.empty:
        return int(exact.iloc[-1]["club_id"])
    partial = all_teams[all_teams["nombre"].str.lower().str.contains(name_lower, regex=False)]
    if not partial.empty:
        return int(partial.iloc[-1]["club_id"])
    return None


def _get_last_stats(club_id: int, n: int = 5) -> Dict[str, float]:
    _load()
    local_rows = _matches_df[_matches_df["club_local_id"] == club_id][[
        "fecha", "pts_local", "pts_visitante", "ganador"
    ]].rename(columns={"pts_local": "pts_for", "pts_visitante": "pts_against"})
    local_rows["win"] = (_matches_df.loc[local_rows.index, "ganador"] == "local").astype(int)

    visit_rows = _matches_df[_matches_df["club_visitante_id"] == club_id][[
        "fecha", "pts_visitante", "pts_local", "ganador"
    ]].rename(columns={"pts_visitante": "pts_for", "pts_local": "pts_against"})
    visit_rows["win"] = (_matches_df.loc[visit_rows.index, "ganador"] == "visitante").astype(int)

    history = pd.concat([local_rows, visit_rows]).sort_values("fecha").tail(n)
    if history.empty:
        return {}

    pts_for = history["pts_for"].mean()
    pts_against = history["pts_against"].mean()
    win_rate = history["win"].mean()
    streak = history["win"].map({1: 1, 0: -1}).sum()
    pt_diff = pts_for - pts_against

    return {
        "pts_avg": pts_for,
        "pts_conceded_avg": pts_against,
        "win_rate": win_rate,
        "streak": streak,
        "pt_diff": pt_diff,
    }


def _get_h2h(home_id: int, away_id: int, window: int = 10) -> Dict:
    _load()
    mask = (
        ((_matches_df["club_local_id"] == home_id) & (_matches_df["club_visitante_id"] == away_id))
        | ((_matches_df["club_local_id"] == away_id) & (_matches_df["club_visitante_id"] == home_id))
    )
    h2h = _matches_df[mask].sort_values("fecha").tail(window)
    if h2h.empty:
        return {"h2h_home_wins": 0, "h2h_total": 0, "h2h_home_rate": 0.5}
    home_wins = int((
        ((h2h["club_local_id"] == home_id) & (h2h["ganador"] == "local")) |
        ((h2h["club_visitante_id"] == home_id) & (h2h["ganador"] == "visitante"))
    ).sum())
    total = len(h2h)
    return {
        "h2h_home_wins": home_wins,
        "h2h_total": total,
        "h2h_home_rate": round(home_wins / total, 4),
    }


def predict(equipo_local: str, equipo_visitante: str) -> Dict:
    _load()
    home_id = _get_club_id(equipo_local)
    away_id = _get_club_id(equipo_visitante)
    if home_id is None:
        raise ValueError(f"Equipo BBL no encontrado: '{equipo_local}'")
    if away_id is None:
        raise ValueError(f"Equipo BBL no encontrado: '{equipo_visitante}'")
    if home_id == away_id:
        raise ValueError("Los dos equipos deben ser distintos.")

    hs5 = _get_last_stats(home_id, 5)
    as5 = _get_last_stats(away_id, 5)
    hs10 = _get_last_stats(home_id, 10)
    as10 = _get_last_stats(away_id, 10)
    hs3 = _get_last_stats(home_id, 3)
    as3 = _get_last_stats(away_id, 3)
    h2h = _get_h2h(home_id, away_id)

    # ELO actual
    home_elo = _elo_ratings.get(home_id, ELO_INITIAL) if _elo_ratings else ELO_INITIAL
    away_elo = _elo_ratings.get(away_id, ELO_INITIAL) if _elo_ratings else ELO_INITIAL

    # Días de descanso desde último partido hasta hoy
    today = pd.Timestamp(date.today(), tz="UTC")
    def days_rest(club_id: int) -> int:
        if _last_game_date and club_id in _last_game_date:
            d = (today - _last_game_date[club_id]).days
            return min(d, MAX_DAYS_REST)
        return 7

    home_rest = days_rest(home_id)
    away_rest = days_rest(away_id)

    def s(stats: Dict, key: str) -> float:
        return stats.get(key, np.nan)

    features = {
        "home_pts_avg_5": s(hs5, "pts_avg"),
        "home_pts_conceded_avg_5": s(hs5, "pts_conceded_avg"),
        "home_win_rate_5": s(hs5, "win_rate"),
        "away_pts_avg_5": s(as5, "pts_avg"),
        "away_pts_conceded_avg_5": s(as5, "pts_conceded_avg"),
        "away_win_rate_5": s(as5, "win_rate"),
        "pts_diff_5": s(hs5, "pts_avg") - s(as5, "pts_avg"),
        "win_rate_diff_5": s(hs5, "win_rate") - s(as5, "win_rate"),
        "home_pts_avg_10": s(hs10, "pts_avg"),
        "home_pts_conceded_avg_10": s(hs10, "pts_conceded_avg"),
        "home_win_rate_10": s(hs10, "win_rate"),
        "away_pts_avg_10": s(as10, "pts_avg"),
        "away_pts_conceded_avg_10": s(as10, "pts_conceded_avg"),
        "away_win_rate_10": s(as10, "win_rate"),
        "pts_diff_10": s(hs10, "pts_avg") - s(as10, "pts_avg"),
        "win_rate_diff_10": s(hs10, "win_rate") - s(as10, "win_rate"),
        "home_streak_5": s(hs5, "streak"),
        "away_streak_5": s(as5, "streak"),
        "home_pt_diff_avg_5": s(hs5, "pt_diff"),
        "away_pt_diff_avg_5": s(as5, "pt_diff"),
        "pt_diff_diff_5": s(hs5, "pt_diff") - s(as5, "pt_diff"),
        "h2h_home_wins": h2h["h2h_home_wins"],
        "h2h_total": h2h["h2h_total"],
        "h2h_home_rate": h2h["h2h_home_rate"],
        "is_home": 1,
        "home_win_rate_3": s(hs3, "win_rate"),
        "away_win_rate_3": s(as3, "win_rate"),
        "win_rate_diff_3": s(hs3, "win_rate") - s(as3, "win_rate"),
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_days_rest": home_rest,
        "away_days_rest": away_rest,
        "rest_diff": home_rest - away_rest,
    }

    feature_cols = _meta["feature_cols"]
    X = pd.DataFrame([{c: features.get(c, np.nan) for c in feature_cols}])

    prob_local = float(_model.predict_proba(X)[0, 1])
    prob_visit = 1.0 - prob_local
    prediccion = "local" if prob_local >= 0.5 else "visitante"
    conf_val = max(prob_local, prob_visit)
    confianza = "alta" if conf_val >= 0.65 else ("media" if conf_val >= 0.55 else "baja")

    teams = {t["club_id"]: t["nombre"] for t in get_available_teams()}
    nombre_local = teams.get(home_id, str(equipo_local))
    nombre_visit = teams.get(away_id, str(equipo_visitante))

    return {
        "equipo_local": nombre_local,
        "equipo_visitante": nombre_visit,
        "prob_local": round(prob_local, 4),
        "prob_visitante": round(prob_visit, 4),
        "prediccion": prediccion,
        "confianza": confianza,
        "features_usadas": {
            "home_win_rate_reciente": round(s(hs5, "win_rate"), 3),
            "away_win_rate_reciente": round(s(as5, "win_rate"), 3),
            "home_pts_promedio": round(s(hs5, "pts_avg"), 1),
            "away_pts_promedio": round(s(as5, "pts_avg"), 1),
            "home_val_promedio": round(s(hs5, "pt_diff"), 1),
            "away_val_promedio": round(s(as5, "pt_diff"), 1),
            "h2h_enfrentamientos": h2h["h2h_total"],
            "h2h_tasa_local": h2h["h2h_home_rate"],
        },
    }
