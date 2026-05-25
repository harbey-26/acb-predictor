"""
Feature Engineering para la BBL (Basketball Bundesliga).

Genera data/processed/bbl_features.csv con features rolling basadas en
puntos y resultados (win/loss). Sin t2/t3/reb/ast (obfuscados en fuente).

Uso:
    python -m model.bbl_feature_engineering
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_PATH = os.path.join(BASE_DIR, "data", "processed", "bbl_matches.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "bbl_features.csv")

WINDOWS = [5, 10]
H2H_WINDOW = 10

# ── ELO constants ──────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K = 20
ELO_HOME_ADVANTAGE = 100   # puntos extra para equipo local en cálculo de probabilidad
ELO_SEASON_REGRESSION = 1 / 3  # regresión hacia la media entre temporadas
MAX_DAYS_REST = 30  # cap para días de descanso (inicio de temporada)


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df = df.sort_values("fecha", kind="mergesort").reset_index(drop=True)
    return df


def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Construye historial largo (una fila por partido por equipo)."""
    local = df[["game_id", "fecha", "club_local_id",
                "pts_local", "pts_visitante", "ganador"]].copy()
    local = local.rename(columns={
        "club_local_id": "club_id",
        "pts_local": "pts_for",
        "pts_visitante": "pts_against",
    })
    local["win"] = (local["ganador"] == "local").astype(int)

    visit = df[["game_id", "fecha", "club_visitante_id",
                "pts_visitante", "pts_local", "ganador"]].copy()
    visit = visit.rename(columns={
        "club_visitante_id": "club_id",
        "pts_visitante": "pts_for",
        "pts_local": "pts_against",
    })
    visit["win"] = (visit["ganador"] == "visitante").astype(int)

    history = pd.concat([local, visit], ignore_index=True)
    history = history.drop(columns=["ganador"])
    history = history.sort_values(["club_id", "fecha", "game_id"], kind="mergesort")
    return history.reset_index(drop=True)


def compute_rolling_stats(history: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling stats por equipo sin data leakage (shift(1))."""
    stat_cols = ["pts_for", "pts_against", "win"]
    results: List[pd.DataFrame] = []

    for club_id, grp in history.groupby("club_id", sort=False):
        grp = grp.sort_values("fecha", kind="mergesort").reset_index(drop=True)
        rolled: Dict[str, pd.Series] = {}
        for col in stat_cols:
            rolled[f"{col}_avg_{window}"] = (
                grp[col].shift(1).rolling(window=window, min_periods=1).mean()
            )
        streak_raw = grp["win"].map({1: 1, 0: -1})
        rolled[f"streak_{window}"] = (
            streak_raw.shift(1).rolling(window=window, min_periods=1).sum()
        )
        rolled_df = pd.DataFrame(rolled)
        rolled_df["game_id"] = grp["game_id"].values
        rolled_df["club_id"] = club_id
        results.append(rolled_df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_h2h(df: pd.DataFrame, window: int = H2H_WINDOW) -> pd.DataFrame:
    """H2H features sin data leakage."""
    df_sorted = df.sort_values("fecha", kind="mergesort").reset_index(drop=True)
    pair_history: Dict[Tuple[int, int], List[Tuple]] = {}
    h2h_home_wins: List[int] = []
    h2h_total: List[int] = []
    game_ids: List[str] = []

    for _, row in df_sorted.iterrows():
        home_id = int(row["club_local_id"])
        away_id = int(row["club_visitante_id"])
        key = (min(home_id, away_id), max(home_id, away_id))
        winner_id = home_id if row["ganador"] == "local" else away_id

        hist = pair_history.get(key, [])[-window:]
        total = len(hist)
        wins = sum(1 for (_, w) in hist if w == home_id)

        h2h_home_wins.append(wins)
        h2h_total.append(total)
        game_ids.append(row["game_id"])

        if key not in pair_history:
            pair_history[key] = []
        pair_history[key].append((row["fecha"], winner_id))

    result = pd.DataFrame({
        "game_id": game_ids,
        "h2h_home_wins": h2h_home_wins,
        "h2h_total": h2h_total,
    })
    result["h2h_home_rate"] = result.apply(
        lambda r: r["h2h_home_wins"] / r["h2h_total"] if r["h2h_total"] > 0 else 0.5,
        axis=1,
    )
    return result


def compute_elo(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calcula ELO pre-partido para cada juego (sin data leakage).
    Procesa todos los partidos (regular + playoffs) en orden cronológico
    para tener ratings actualizados, pero devuelve ELO para todos los game_id.
    Regresión hacia la media (1/3) al inicio de cada temporada.
    """
    ratings: Dict[int, float] = {}
    prev_season: Optional[str] = None
    results: Dict[str, Dict] = {}

    for _, row in df.sort_values("fecha", kind="mergesort").iterrows():
        home_id = int(row["club_local_id"])
        away_id = int(row["club_visitante_id"])
        season = row["temporada"]

        # Regresión al inicio de cada temporada nueva
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

        results[row["game_id"]] = {
            "home_elo": round(home_elo, 2),
            "away_elo": round(away_elo, 2),
            "elo_diff": round(home_elo - away_elo, 2),
        }

        # Actualizar ratings post-partido (ventaja local incorporada en expected)
        home_elo_adj = home_elo + ELO_HOME_ADVANTAGE
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo_adj) / 400))
        actual_home = 1.0 if row["ganador"] == "local" else 0.0

        ratings[home_id] = home_elo + ELO_K * (actual_home - exp_home)
        ratings[away_id] = away_elo + ELO_K * ((1 - actual_home) - (1 - exp_home))

    return results


def compute_days_rest(df: pd.DataFrame) -> Dict[str, Dict]:
    """Días de descanso desde el último partido de cada equipo (cap: MAX_DAYS_REST)."""
    last_date: Dict[int, pd.Timestamp] = {}
    results: Dict[str, Dict] = {}

    for _, row in df.sort_values("fecha", kind="mergesort").iterrows():
        home_id = int(row["club_local_id"])
        away_id = int(row["club_visitante_id"])
        game_date = row["fecha"]
        gid = row["game_id"]

        home_rest = int((game_date - last_date[home_id]).days) if home_id in last_date else 7
        away_rest = int((game_date - last_date[away_id]).days) if away_id in last_date else 7

        home_rest = min(home_rest, MAX_DAYS_REST)
        away_rest = min(away_rest, MAX_DAYS_REST)

        results[gid] = {
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
            "rest_diff": home_rest - away_rest,
        }

        last_date[home_id] = game_date
        last_date[away_id] = game_date

    return results


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset de features para partidos de liga regular."""
    df_lr = df[df["is_playoff"] == 0].copy().reset_index(drop=True)
    if df_lr.empty:
        raise ValueError("No hay partidos de liga regular en los datos BBL.")

    # ELO y descanso se calculan sobre TODOS los partidos (incl. playoffs)
    # para tener ratings precisos, pero sólo se usan en juegos de liga regular
    elo_lookup = compute_elo(df)
    rest_lookup = compute_days_rest(df)

    history = build_team_history(df_lr)

    lookups: Dict[int, Dict[Tuple[str, int], Dict]] = {}
    for w in WINDOWS:
        rolled = compute_rolling_stats(history, window=w)
        lookup: Dict[Tuple[str, int], Dict] = {}
        for _, row in rolled.iterrows():
            key = (row["game_id"], int(row["club_id"]))
            lookup[key] = row.drop(["game_id", "club_id"]).to_dict()
        lookups[w] = lookup

    h2h_df = compute_h2h(df_lr)
    h2h_lookup = {
        row["game_id"]: {
            "h2h_home_wins": row["h2h_home_wins"],
            "h2h_total": row["h2h_total"],
            "h2h_home_rate": row["h2h_home_rate"],
        }
        for _, row in h2h_df.iterrows()
    }

    # Window 3 (win rate de los últimos 3 partidos – feature más importante en ACB)
    rolled_3 = compute_rolling_stats(history, window=3)
    lookup_3: Dict[Tuple[str, int], Dict] = {}
    for _, row in rolled_3.iterrows():
        key = (row["game_id"], int(row["club_id"]))
        lookup_3[key] = row.drop(["game_id", "club_id"]).to_dict()

    feature_rows: List[Dict] = []
    for _, match in df_lr.iterrows():
        gid = match["game_id"]
        home_id = int(match["club_local_id"])
        away_id = int(match["club_visitante_id"])

        row: Dict = {
            "game_id": gid,
            "temporada": match["temporada"],
            "fecha": match["fecha"],
            "equipo_local": match["equipo_local"],
            "equipo_visitante": match["equipo_visitante"],
            "club_local_id": home_id,
            "club_visitante_id": away_id,
        }

        for w in WINDOWS:
            hs = lookups[w].get((gid, home_id), {})
            as_ = lookups[w].get((gid, away_id), {})

            row[f"home_pts_avg_{w}"] = hs.get(f"pts_for_avg_{w}", np.nan)
            row[f"home_pts_conceded_avg_{w}"] = hs.get(f"pts_against_avg_{w}", np.nan)
            row[f"home_win_rate_{w}"] = hs.get(f"win_avg_{w}", np.nan)
            row[f"away_pts_avg_{w}"] = as_.get(f"pts_for_avg_{w}", np.nan)
            row[f"away_pts_conceded_avg_{w}"] = as_.get(f"pts_against_avg_{w}", np.nan)
            row[f"away_win_rate_{w}"] = as_.get(f"win_avg_{w}", np.nan)
            row[f"pts_diff_{w}"] = (
                row[f"home_pts_avg_{w}"] - row[f"away_pts_avg_{w}"]
                if not (np.isnan(row.get(f"home_pts_avg_{w}", np.nan))
                        or np.isnan(row.get(f"away_pts_avg_{w}", np.nan)))
                else np.nan
            )
            row[f"win_rate_diff_{w}"] = (
                row[f"home_win_rate_{w}"] - row[f"away_win_rate_{w}"]
                if not (np.isnan(row.get(f"home_win_rate_{w}", np.nan))
                        or np.isnan(row.get(f"away_win_rate_{w}", np.nan)))
                else np.nan
            )

        # Ventana 3
        hs3 = lookup_3.get((gid, home_id), {})
        as3 = lookup_3.get((gid, away_id), {})
        row["home_win_rate_3"] = hs3.get("win_avg_3", np.nan)
        row["away_win_rate_3"] = as3.get("win_avg_3", np.nan)
        row["win_rate_diff_3"] = (
            row["home_win_rate_3"] - row["away_win_rate_3"]
            if not (np.isnan(row.get("home_win_rate_3", np.nan))
                    or np.isnan(row.get("away_win_rate_3", np.nan)))
            else np.nan
        )

        # Racha y pt_diff individuales (ventana 5)
        hs5 = lookups[5].get((gid, home_id), {})
        as5 = lookups[5].get((gid, away_id), {})
        row["home_streak_5"] = hs5.get("streak_5", np.nan)
        row["away_streak_5"] = as5.get("streak_5", np.nan)
        row["home_pt_diff_avg_5"] = (
            hs5.get("pts_for_avg_5", np.nan) - hs5.get("pts_against_avg_5", np.nan)
            if not (np.isnan(hs5.get("pts_for_avg_5", np.nan))
                    or np.isnan(hs5.get("pts_against_avg_5", np.nan)))
            else np.nan
        )
        row["away_pt_diff_avg_5"] = (
            as5.get("pts_for_avg_5", np.nan) - as5.get("pts_against_avg_5", np.nan)
            if not (np.isnan(as5.get("pts_for_avg_5", np.nan))
                    or np.isnan(as5.get("pts_against_avg_5", np.nan)))
            else np.nan
        )
        row["pt_diff_diff_5"] = (
            row["home_pt_diff_avg_5"] - row["away_pt_diff_avg_5"]
            if not (np.isnan(row.get("home_pt_diff_avg_5", np.nan))
                    or np.isnan(row.get("away_pt_diff_avg_5", np.nan)))
            else np.nan
        )

        h2h = h2h_lookup.get(gid, {})
        row["h2h_home_wins"] = h2h.get("h2h_home_wins", 0)
        row["h2h_total"] = h2h.get("h2h_total", 0)
        row["h2h_home_rate"] = h2h.get("h2h_home_rate", 0.5)
        row["is_home"] = 1

        # ELO pre-partido
        elo = elo_lookup.get(gid, {})
        row["home_elo"] = elo.get("home_elo", ELO_INITIAL)
        row["away_elo"] = elo.get("away_elo", ELO_INITIAL)
        row["elo_diff"] = elo.get("elo_diff", 0.0)

        # Días de descanso
        rest = rest_lookup.get(gid, {})
        row["home_days_rest"] = rest.get("home_days_rest", 7)
        row["away_days_rest"] = rest.get("away_days_rest", 7)
        row["rest_diff"] = rest.get("rest_diff", 0)

        row["target"] = 1 if match["ganador"] == "local" else 0
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def main() -> None:
    print("Cargando bbl_matches.csv...")
    df = load_matches()
    total = len(df)
    regular = (df["is_playoff"] == 0).sum()
    print(f"  Total: {total} partidos | Liga regular: {regular}")

    print("Construyendo features BBL...")
    features_df = build_features(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    n_rows, n_cols = features_df.shape
    target_rate = features_df["target"].mean() * 100
    meta_cols = {"game_id", "temporada", "fecha", "equipo_local", "equipo_visitante",
                 "club_local_id", "club_visitante_id", "target"}
    feat_cols = [c for c in features_df.columns if c not in meta_cols]
    print(f"Dataset: {n_rows} filas, {len(feat_cols)} features")
    print(f"Target=1 (local gana): {target_rate:.1f}%")
    print(f"Guardado en: {OUTPUT_PATH}")
    by_season = features_df.groupby("temporada").size()
    print("Por temporada:")
    for s, c in by_season.items():
        print(f"  {s}: {c} partidos")


if __name__ == "__main__":
    main()
