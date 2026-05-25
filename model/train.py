"""
Fase 3: Entrenamiento del modelo predictivo ACB.

Estrategia de validación temporal:
  - Train : temporadas 2020-21, 2021-22, 2022-23
  - Val   : temporada 2023-24  (para ajuste de hiperparámetros)
  - Test  : temporada 2024-25  (evaluación final, nunca tocada durante entrenamiento)

Uso:
    python -m model.train
    python -m model.train --no-tune    # salta búsqueda de hiperparámetros
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "model.pkl")
META_PATH     = os.path.join(ARTIFACTS_DIR, "model_meta.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")

# ---------------------------------------------------------------------------
# Columnas de features (las 33 numéricas, excluyendo metadatos y target)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "home_pts_avg_5", "home_pts_conceded_avg_5", "home_win_rate_5",
    "away_pts_avg_5", "away_pts_conceded_avg_5", "away_win_rate_5",
    "pts_diff_5", "win_rate_diff_5",
    "home_pts_avg_10", "home_pts_conceded_avg_10", "home_win_rate_10",
    "away_pts_avg_10", "away_pts_conceded_avg_10", "away_win_rate_10",
    "pts_diff_10", "win_rate_diff_10",
    "home_t3_pct_avg_5", "home_t2_pct_avg_5", "home_reb_avg_5",
    "home_ast_avg_5", "home_val_avg_5",
    "away_t3_pct_avg_5", "away_t2_pct_avg_5", "away_reb_avg_5",
    "away_ast_avg_5", "away_val_avg_5",
    "val_diff_5",
    "home_streak_5", "away_streak_5",
    "h2h_home_wins", "h2h_total", "h2h_home_rate",
    "is_home",
]

TARGET_COL = "target"

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23"]
VAL_SEASON    = "2023-24"
TEST_SEASON   = "2024-25"

# ---------------------------------------------------------------------------
# Carga y splits
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    return df


def split_data(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    train = df[df["temporada"].isin(TRAIN_SEASONS)]
    val   = df[df["temporada"] == VAL_SEASON]
    test  = df[df["temporada"] == TEST_SEASON]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_val   = val[FEATURE_COLS]
    y_val   = val[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Pipeline: imputer → scaler → XGBoost
# ---------------------------------------------------------------------------

def build_pipeline(params: Optional[Dict] = None) -> Pipeline:
    """
    Construye un sklearn Pipeline con:
      1. SimpleImputer (mediana) — gestiona el ~1% de NaN en primeras jornadas
      2. StandardScaler — XGBoost no lo necesita pero mejora la calibración
      3. XGBClassifier
    """
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("xgb",     XGBClassifier(**default_params)),
    ])


# ---------------------------------------------------------------------------
# Búsqueda de hiperparámetros (grid search temporal)
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.03, 0.05, 0.1],
    "n_estimators":     [200, 300, 500],
    "min_child_weight": [1, 3, 5],
    "subsample":        [0.7, 0.8],
}

def tune_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> Dict:
    """
    Búsqueda en grid usando validación temporal (train → val).
    Optimiza ROC-AUC en el conjunto de validación.
    """
    print("Búsqueda de hiperparámetros...")
    best_auc  = 0.0
    best_params: Dict = {}
    total = len(list(ParameterGrid(PARAM_GRID)))

    for i, params in enumerate(ParameterGrid(PARAM_GRID), 1):
        pipe = build_pipeline(params)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        if auc > best_auc:
            best_auc    = auc
            best_params = params
        if i % 20 == 0:
            print(f"  {i}/{total} — mejor AUC val: {best_auc:.4f}")

    print(f"  → Mejor AUC val: {best_auc:.4f} con {best_params}")
    return best_params


# ---------------------------------------------------------------------------
# Calibración de probabilidades (Platt scaling)
# ---------------------------------------------------------------------------

def calibrate_model(pipeline: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series) -> CalibratedClassifierCV:
    """
    Envuelve el pipeline con calibración Platt (sigmoid).
    Usa el conjunto de validación como datos de calibración.
    """
    calibrated = CalibratedClassifierCV(pipeline, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate(model, X: pd.DataFrame, y: pd.Series, split_name: str) -> Dict:
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "split":        split_name,
        "n":            len(y),
        "accuracy":     round(accuracy_score(y, pred),   4),
        "roc_auc":      round(roc_auc_score(y, proba),   4),
        "brier_score":  round(brier_score_loss(y, proba), 4),
        "log_loss":     round(log_loss(y, proba),         4),
    }
    return metrics


def print_metrics(metrics: Dict) -> None:
    print(f"\n  [{metrics['split']}]  n={metrics['n']}")
    print(f"    Accuracy   : {metrics['accuracy']:.1%}")
    print(f"    ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"    Brier score: {metrics['brier_score']:.4f}  (↓ mejor, 0=perfecto)")
    print(f"    Log-loss   : {metrics['log_loss']:.4f}")


def save_feature_importance(pipeline: Pipeline) -> None:
    """Guarda importancia de features del XGBoost (antes de calibración)."""
    xgb_model = pipeline.named_steps["xgb"]
    importance = xgb_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print("\n  Top 10 features más importantes:")
    for _, row in fi_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<30} {row['importance']:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tune: bool = True) -> None:
    print("=" * 60)
    print("FASE 3 — ENTRENAMIENTO DEL MODELO ACB")
    print("=" * 60)

    # 1. Datos
    print("\n[1/5] Cargando features...")
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    print(f"  Train: {len(X_train)} partidos ({', '.join(TRAIN_SEASONS)})")
    print(f"  Val  : {len(X_val)} partidos ({VAL_SEASON})")
    print(f"  Test : {len(X_test)} partidos ({TEST_SEASON})")

    # 2. Hiperparámetros
    if tune:
        print("\n[2/5] Ajuste de hiperparámetros...")
        best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)
    else:
        best_params = {}
        print("\n[2/5] Sin ajuste — usando parámetros por defecto.")

    # 3. Entrenamiento final en train+val
    print("\n[3/5] Entrenando modelo final (train + val)...")
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_pipeline = build_pipeline(best_params)
    final_pipeline.fit(X_trainval, y_trainval)

    # 4. Calibración con el conjunto de validación
    print("\n[4/5] Calibrando probabilidades (Platt scaling con val)...")
    # Reentrenamos en train solo y calibramos con val para evitar sobreajuste
    base_pipeline = build_pipeline(best_params)
    base_pipeline.fit(X_train, y_train)
    calibrated_model = calibrate_model(base_pipeline, X_val, y_val)

    # 5. Evaluación
    print("\n[5/5] Evaluación:")
    all_metrics = []
    for name, X_ev, y_ev in [
        ("TRAIN",  X_train, y_train),
        ("VAL",    X_val,   y_val),
        ("TEST",   X_test,  y_test),
    ]:
        m = evaluate(calibrated_model, X_ev, y_ev, name)
        print_metrics(m)
        all_metrics.append(m)

    # Importancia de features (del pipeline base, antes de calibración)
    save_feature_importance(base_pipeline)

    # 6. Guardar artefactos
    print("\n  Guardando artefactos...")
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"    Modelo    → {MODEL_PATH}")

    meta = {
        "feature_cols":   FEATURE_COLS,
        "train_seasons":  TRAIN_SEASONS,
        "val_season":     VAL_SEASON,
        "test_season":    TEST_SEASON,
        "best_params":    best_params,
        "metrics":        all_metrics,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"    Metadatos → {META_PATH}")

    print("\n" + "=" * 60)
    test_m = next(m for m in all_metrics if m["split"] == "TEST")
    print(f"✓ Resultado final en TEST (temporada {TEST_SEASON}):")
    print(f"  Accuracy {test_m['accuracy']:.1%}  |  ROC-AUC {test_m['roc_auc']:.4f}  |  Brier {test_m['brier_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", dest="tune", action="store_false",
                        help="Salta búsqueda de hiperparámetros")
    args = parser.parse_args()
    main(tune=args.tune)
