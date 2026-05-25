"""
Entrenamiento del modelo BBL (Basketball Bundesliga).

Temporal split:
  Train: 2020-21, 2021-22, 2022-23
  Val (calibración Platt): 2023-24
  Test: 2024-25

Uso:
    python -m model.bbl_train
    python -m model.bbl_train --no-tune   # sin búsqueda de hiperparámetros
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "bbl_features.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "bbl_model.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "bbl_model_meta.json")

_BOX_STATS  = ["fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"]
_BOX_WINDOWS = [5, 10]
# Solo diferencias (home - away) por ventana: 6 stats × 2 ventanas = 12 features
# Mejor generalización que incluir home_/away_ por separado (menos overfitting)
_BOX_FEATURES = [
    f"{stat}_diff_{w}"
    for w in _BOX_WINDOWS
    for stat in _BOX_STATS
]

FEATURE_COLS = [
    "home_pts_avg_5", "home_pts_conceded_avg_5", "home_win_rate_5",
    "away_pts_avg_5", "away_pts_conceded_avg_5", "away_win_rate_5",
    "pts_diff_5", "win_rate_diff_5",
    "home_pts_avg_10", "home_pts_conceded_avg_10", "home_win_rate_10",
    "away_pts_avg_10", "away_pts_conceded_avg_10", "away_win_rate_10",
    "pts_diff_10", "win_rate_diff_10",
    "home_streak_5", "away_streak_5",
    "home_pt_diff_avg_5", "away_pt_diff_avg_5", "pt_diff_diff_5",
    "h2h_home_wins", "h2h_total", "h2h_home_rate",
    "is_home",
    "home_win_rate_3", "away_win_rate_3", "win_rate_diff_3",
    # v2: ELO dinámico + días de descanso
    "home_elo", "away_elo", "elo_diff",
    "home_days_rest", "away_days_rest", "rest_diff",
    # v3: boxscore FlashScore — diferencias (home - away) en FG%, 3P%, FT%, REB, AST, TOV
    *_BOX_FEATURES,
]

TARGET_COL = "target"
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23"]
VAL_SEASON = "2023-24"
TEST_SEASON = "2024-25"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    return df


def split(df: pd.DataFrame) -> tuple:
    train = df[df["temporada"].isin(TRAIN_SEASONS)]
    val = df[df["temporada"] == VAL_SEASON]
    test = df[df["temporada"] == TEST_SEASON]
    return train, val, test


def make_pipeline(xgb_params: Optional[Dict] = None) -> Pipeline:
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(**xgb_params)),
    ])


def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    tune: bool = True,
) -> tuple:
    X_tr = train[FEATURE_COLS]
    y_tr = train[TARGET_COL]
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET_COL]
    X_te = test[FEATURE_COLS]
    y_te = test[TARGET_COL]

    if tune:
        print("  Buscando hiperparámetros (GridSearch)...")
        param_grid = {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [3, 4],
            "clf__learning_rate": [0.03, 0.05],
            "clf__subsample": [0.7, 0.8],
            "clf__min_child_weight": [2, 3],
        }
        base_pipe = make_pipeline()
        gs = GridSearchCV(
            base_pipe, param_grid, cv=3, scoring="accuracy",
            n_jobs=-1, verbose=0,
        )
        gs.fit(X_tr, y_tr)
        best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        base_xgb_params = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        base_xgb_params.update(best_params)
        pipe = make_pipeline(base_xgb_params)
    else:
        pipe = make_pipeline()

    pipe.fit(X_tr, y_tr)

    # Platt calibration en val
    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)

    # Evaluar en test
    y_pred = calibrated.predict(X_te)
    y_proba = calibrated.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    brier = brier_score_loss(y_te, y_proba)

    return calibrated, acc, auc, brier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    print("=== Entrenamiento modelo BBL ===")
    print(f"Features path: {FEATURES_PATH}")

    df = load_data()
    print(f"Dataset: {len(df)} partidos")
    by_season = df.groupby("temporada").size()
    for s, c in by_season.items():
        print(f"  {s}: {c}")

    available = set(df["temporada"].unique())
    if TEST_SEASON not in available:
        print(f"\n[WARN] Temporada de test {TEST_SEASON} no disponible.")
        print("       Entrenando con lo disponible (sin evaluación final).")

    train, val, test = split(df)
    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    if len(test) == 0:
        print("[ERROR] No hay datos de test. Ejecuta el scraper para 2024-25.")
        return

    tune = not args.no_tune
    print(f"\nEntrenando XGBoost {'con' if tune else 'sin'} hyperparameter tuning...")

    model, acc, auc, brier = train_and_evaluate(train, val, test, tune=tune)

    print(f"\n{'='*50}")
    print(f"  Accuracy test  : {acc*100:.1f}%")
    print(f"  ROC-AUC test   : {auc:.4f}")
    print(f"  Brier score    : {brier:.4f}")
    print(f"{'='*50}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")

    meta = {
        "model_name": "BBL_XGBoost_v3_boxscore",
        "accuracy_test": round(acc, 4),
        "roc_auc_test": round(auc, 4),
        "brier_test": round(brier, 4),
        "feature_cols": FEATURE_COLS,
        "train_seasons": TRAIN_SEASONS,
        "val_season": VAL_SEASON,
        "test_season": TEST_SEASON,
        "n_features": len(FEATURE_COLS),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta guardada en: {META_PATH}")


# Make Optional importable
from typing import Optional

if __name__ == "__main__":
    main()
