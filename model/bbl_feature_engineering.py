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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset de features para partidos de liga regular."""
    df_lr = df[df["is_playoff"] == 0].copy().reset_index(drop=True)
    if df_lr.empty:
        raise ValueError("No hay partidos de liga regular en los datos BBL.")

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
