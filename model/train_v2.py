"""
train_v2.py — Experimentos para mejorar el baseline 69.3% (2024-25 test).

Experimentos:
  A) XGBoost baseline replicado
  B) XGBoost con más datos (train incluye 2023-24)
  C) LightGBM con split original
  D) LightGBM con más datos
  E) Ensemble (media de probas XGB + LGB, más datos)

Uso:
    python -m model.train_v2          # muestra tabla, no guarda
    python -m model.train_v2 --deploy # guarda el mejor si supera baseline
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] LightGBM no instalado — se omiten experimentos LGB")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "model.pkl")
BACKUP_PATH   = os.path.join(ARTIFACTS_DIR, "model_backup.pkl")
META_PATH     = os.path.join(ARTIFACTS_DIR, "model_meta.json")

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

SEASONS_ALL  = ["2020-21", "2021-22", "2022-23", "2023-24"]
TRAIN_ORIG   = ["2020-21", "2021-22", "2022-23"]
VAL_SEASON   = "2023-24"
TEST_SEASON  = "2024-25"
BASELINE_ACC = 0.693


def load_df():
    df = pd.read_csv(FEATURES_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    return df


def get_splits(df, train_seasons):
    train = df[df["temporada"].isin(train_seasons)]
    test  = df[df["temporada"] == TEST_SEASON]
    return (train[FEATURE_COLS], train[TARGET_COL],
            test[FEATURE_COLS],  test[TARGET_COL])


def xgb_pipe(params=None):
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
             gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
             eval_metric="logloss", random_state=42, n_jobs=-1)
    if params:
        p.update(params)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()),
                     ("clf", XGBClassifier(**p))])


def lgb_pipe(params=None):
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             num_leaves=31, subsample=0.8, colsample_bytree=0.8,
             min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)
    if params:
        p.update(params)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()),
                     ("clf", LGBMClassifier(**p))])


def calibrate_prefit(pipe, X_cal, y_cal):
    cal = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")
    cal.fit(X_cal, y_cal)
    return cal


def calibrate_cv(pipe, X_tr, y_tr, cv=5):
    cal = CalibratedClassifierCV(pipe, cv=cv, method="sigmoid")
    cal.fit(X_tr, y_tr)
    return cal


def eval_model(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        "n":        len(y),
        "accuracy": round(accuracy_score(y, pred),   4),
        "roc_auc":  round(roc_auc_score(y, proba),   4),
        "brier":    round(brier_score_loss(y, proba), 4),
    }


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X) for m in self.models])
        return probas.mean(axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run_experiments(df):
    results = []

    X_tr3, y_tr3, X_te, y_te = get_splits(df, TRAIN_ORIG)
    X_tr4, y_tr4, _,   _     = get_splits(df, SEASONS_ALL)
    val = df[df["temporada"] == VAL_SEASON]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]

    # A: XGBoost baseline
    print("  [A] XGBoost baseline (3 temporadas + Platt val)...")
    p = xgb_pipe(); p.fit(X_tr3, y_tr3)
    m_a = calibrate_prefit(p, X_val, y_val)
    r_a = eval_model(m_a, X_te, y_te)
    results.append({"name": "A_XGB_3seasons", "model": m_a, **r_a})
    print(f"       Acc={r_a['accuracy']:.1%}  AUC={r_a['roc_auc']:.4f}  Brier={r_a['brier']:.4f}")

    # B: XGBoost 4 temporadas, calibración CV-5
    print("  [B] XGBoost 4 temporadas (CV-5 calibración)...")
    m_b = calibrate_cv(xgb_pipe(), X_tr4, y_tr4, cv=5)
    r_b = eval_model(m_b, X_te, y_te)
    results.append({"name": "B_XGB_4seasons", "model": m_b, **r_b})
    print(f"       Acc={r_b['accuracy']:.1%}  AUC={r_b['roc_auc']:.4f}  Brier={r_b['brier']:.4f}")

    if HAS_LGB:
        # C: LightGBM 3 temporadas
        print("  [C] LightGBM 3 temporadas + Platt val...")
        p = lgb_pipe(); p.fit(X_tr3, y_tr3)
        m_c = calibrate_prefit(p, X_val, y_val)
        r_c = eval_model(m_c, X_te, y_te)
        results.append({"name": "C_LGB_3seasons", "model": m_c, **r_c})
        print(f"       Acc={r_c['accuracy']:.1%}  AUC={r_c['roc_auc']:.4f}  Brier={r_c['brier']:.4f}")

        # D: LightGBM 4 temporadas
        print("  [D] LightGBM 4 temporadas (CV-5 calibración)...")
        m_d = calibrate_cv(lgb_pipe(), X_tr4, y_tr4, cv=5)
        r_d = eval_model(m_d, X_te, y_te)
        results.append({"name": "D_LGB_4seasons", "model": m_d, **r_d})
        print(f"       Acc={r_d['accuracy']:.1%}  AUC={r_d['roc_auc']:.4f}  Brier={r_d['brier']:.4f}")

        # E: Ensemble XGB(B) + LGB(D)
        print("  [E] Ensemble XGB+LGB 4 temporadas...")
        m_e = EnsembleModel([m_b, m_d])
        r_e = eval_model(m_e, X_te, y_te)
        results.append({"name": "E_Ensemble", "model": m_e, **r_e})
        print(f"       Acc={r_e['accuracy']:.1%}  AUC={r_e['roc_auc']:.4f}  Brier={r_e['brier']:.4f}")

    return results


def main(deploy=False):
    print("=" * 60)
    print("TRAIN V2 — Comparativa de modelos ACB")
    print(f"Baseline: {BASELINE_ACC:.1%} accuracy en test 2024-25")
    print("=" * 60)

    df = load_df()
    print(f"\nDataset: {len(df)} filas | test={len(df[df['temporada']==TEST_SEASON])} partidos\n")

    results = run_experiments(df)

    print("\n" + "=" * 60)
    print(f"{'Modelo':<23} {'Accuracy':>9} {'ROC-AUC':>9} {'Brier':>7} {'vs baseline':>12}")
    print("-" * 60)
    best_acc  = BASELINE_ACC
    best_result = None
    for r in results:
        delta = r["accuracy"] - BASELINE_ACC
        flag  = "  ← MEJOR" if r["accuracy"] > best_acc else ""
        if r["accuracy"] > best_acc:
            best_acc    = r["accuracy"]
            best_result = r
        print(f"  {r['name']:<21} {r['accuracy']:>8.1%} {r['roc_auc']:>9.4f} "
              f"{r['brier']:>7.4f}   {delta:>+.1%}{flag}")
    print("-" * 60)
    print(f"  {'BASELINE (actual)':<21} {BASELINE_ACC:>8.1%}   (referencia)")
    print("=" * 60)

    if best_result is None:
        print("\n⚠️  Ningún modelo supera el baseline. Modelo actual conservado.")
        return

    print(f"\n✓ Mejor: {best_result['name']}  "
          f"Acc={best_result['accuracy']:.1%}  "
          f"AUC={best_result['roc_auc']:.4f}  "
          f"Brier={best_result['brier']:.4f}")

    if deploy:
        print("\n  Guardando modelo mejorado...")
        if os.path.exists(MODEL_PATH):
            import shutil
            shutil.copy(MODEL_PATH, BACKUP_PATH)
            print(f"  Backup: {BACKUP_PATH}")
        joblib.dump(best_result["model"], MODEL_PATH)
        print(f"  Modelo: {MODEL_PATH}")

        meta = {
            "model_name":        best_result["name"],
            "accuracy_test":     best_result["accuracy"],
            "roc_auc_test":      best_result["roc_auc"],
            "brier_test":        best_result["brier"],
            "baseline_accuracy": BASELINE_ACC,
            "improvement":       round(best_result["accuracy"] - BASELINE_ACC, 4),
            "feature_cols":      FEATURE_COLS,
            "train_seasons":     SEASONS_ALL,
            "test_season":       TEST_SEASON,
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  Metadatos: {META_PATH}")
        print("\n✓ Despliegue completado.")
    else:
        print("\n  Ejecuta con --deploy para guardar el mejor modelo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()
    main(deploy=args.deploy)
