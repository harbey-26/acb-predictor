# Handoff para Gemini — Predictor de Baloncesto

> Fecha: 2026-05-25  
> Repo: https://github.com/harbey-26/acb-predictor  
> Deploy live: https://perpetual-encouragement-production-bd2c.up.railway.app  
> Stack: Python 3.9 · FastAPI · XGBoost · Railway

---

## Contexto del proyecto

App de predicción de partidos de baloncesto con dos ligas:
- **ACB** (Liga española) — modelo XGBoost v2, 69.9% accuracy ✅ completo
- **BBL** (Bundesliga alemana) — modelo XGBoost v3 recién terminado, AUC 0.660

Se acaba de completar el pipeline BBL v3: scraped 1,631 boxscores de FlashScore,
enriquecido el CSV de partidos, reentrenado el modelo con 46 features.

---

## Lo que falta — por orden de prioridad

---

### TAREA 1 🔴 Fix de latencia en predictor BBL (155 ms → <20 ms)

**Archivo:** `model/bbl_predictor.py`

**Problema:** La función `_get_last_boxscore_stats(club_id, n)` (línea ~186)
itera con `iterrows()` sobre todas las 1,630 filas del CSV **4 veces por predicción**
(hbox5, abox5, hbox10, abox10). Esto causa 155 ms de latencia.

**Fix a implementar:** Construir el índice una sola vez en `_load()`.

Cambio 1 — añadir variable global al inicio del archivo (junto a las otras):
```python
_boxscore_lookup: Optional[Dict[int, pd.DataFrame]] = None
```

Cambio 2 — añadir función de construcción del índice:
```python
def _build_boxscore_index() -> None:
    """Preconstruye índice club_id → DataFrame de sus actuaciones de boxscore."""
    global _boxscore_lookup
    if "home_fg_pct" not in _matches_df.columns:
        _boxscore_lookup = {}
        return

    box_map = {
        "fg_pct":  ("home_fg_pct",  "away_fg_pct"),
        "fg3_pct": ("home_fg3_pct", "away_fg3_pct"),
        "ft_pct":  ("home_ft_pct",  "away_ft_pct"),
        "reb":     ("home_reb",     "away_reb"),
        "ast":     ("home_ast",     "away_ast"),
        "tov":     ("home_tov",     "away_tov"),
    }

    from collections import defaultdict
    club_rows = defaultdict(list)

    for _, row in _matches_df.iterrows():
        fecha = row["fecha"]
        home_id = int(row["club_local_id"])
        away_id = int(row["club_visitante_id"])

        home_entry = {"fecha": fecha}
        away_entry = {"fecha": fecha}
        for stat, (hc, ac) in box_map.items():
            home_entry[stat] = row.get(hc, np.nan)
            away_entry[stat] = row.get(ac, np.nan)

        club_rows[home_id].append(home_entry)
        club_rows[away_id].append(away_entry)

    _boxscore_lookup = {
        club_id: pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)
        for club_id, rows in club_rows.items()
    }
```

Cambio 3 — al final de `_load()`, llamar al índice:
```python
def _load() -> None:
    global _model, _meta, _matches_df
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _meta is None:
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    if _matches_df is None:
        path = ENRICHED_MATCHES_PATH if os.path.exists(ENRICHED_MATCHES_PATH) else MATCHES_PATH
        _matches_df = pd.read_csv(path)
        _matches_df["fecha"] = pd.to_datetime(_matches_df["fecha"], utc=True)
        _compute_elo_and_rest()
        _build_boxscore_index()   # <-- añadir esta línea
```

Cambio 4 — reescribir `_get_last_boxscore_stats` para usar el índice:
```python
def _get_last_boxscore_stats(club_id: int, n: int = 5) -> Dict[str, float]:
    _load()
    if not _boxscore_lookup or club_id not in _boxscore_lookup:
        return {}
    hist = _boxscore_lookup[club_id].tail(n)
    result = {}
    for stat in BOX_STATS:
        if stat in hist.columns:
            val = hist[stat].dropna().mean()
            result[stat] = float(val) if not np.isnan(val) else np.nan
    return result
```

**Verificación:** Después del cambio, la latencia debe bajar de 155 ms a <20 ms:
```bash
python3 -c "
import time
from model.bbl_predictor import predict
predict('ALBA Berlin', 'Ratiopharm Ulm')  # warm up
t0 = time.time()
for _ in range(10):
    predict('ALBA Berlin', 'FC Bayern Muenchen Basketball')
print(f'Latencia promedio: {(time.time()-t0)/10*1000:.1f}ms')
"
```

---

### TAREA 2 🟡 Mapear Baskets Paderborn en FlashScore (cobertura 2023-24: 89.6%)

**Archivo:** `scraper/bbl_merge_boxscores.py`

Primero, encontrar el nombre exacto en los datos de FlashScore:
```bash
python3 -c "
import json
idx = json.load(open('data/flashscore/index/2023-24_index.json'))
all_names = set()
for r in idx:
    all_names.add(r['fs_home_raw'])
    all_names.add(r['fs_away_raw'])
print(sorted(all_names))
"
```

Luego añadir la entrada al dict `FS_TO_CSV` en `bbl_merge_boxscores.py` (línea ~41).
Paderborn debería aparecer como algo como `"Paderborn"` o `"Baskets Paderborn"`.

Después re-ejecutar el pipeline completo:
```bash
python3 -m scraper.bbl_merge_boxscores
python3 -m model.bbl_feature_engineering
python3 -m model.bbl_train
```

---

### TAREA 3 ⚪ Calibración Platt para BBL (mejora confiabilidad de probabilidades)

**Archivo:** `model/bbl_train.py`

El modelo ACB usa `CalibratedClassifierCV`. El BBL no. Añadir al final de
`train_and_evaluate()`, después del fit principal, usar val_season para calibrar:

```python
from sklearn.calibration import CalibratedClassifierCV

# Después de encontrar el mejor pipeline (al final del GridSearch):
calibrated = CalibratedClassifierCV(best_pipeline, method='sigmoid', cv='prefit')
calibrated.fit(X_val, y_val)
# Devolver calibrated en lugar de best_pipeline
```

---

### TAREA 4 ⚪ Métricas BBL (AUC 0.660, objetivo era 0.68-0.70)

Si las tareas 1-3 están completas, intentar mejorar el modelo:

**Opción A** — añadir features individuales (no solo diffs) para 3P%:
En `model/bbl_train.py`, añadir a `_BOX_FEATURES`:
```python
"home_fg3_pct_avg_5", "away_fg3_pct_avg_5",
"home_fg3_pct_avg_10", "away_fg3_pct_avg_10",
```

**Opción B** — añadir ventana de 3 partidos:
```python
_BOX_WINDOWS = [3, 5, 10]
```
(requiere también añadir la ventana en `bbl_feature_engineering.py`)

En ambos casos, re-ejecutar `bbl_feature_engineering.py` y `bbl_train.py`,
y comparar AUC test. Solo guardar si mejora.

---

## Cómo entregar el trabajo de vuelta a Claude

Cuando termines las tareas, hacer commit y push:
```bash
git add model/bbl_predictor.py model/bbl_train.py scraper/bbl_merge_boxscores.py
git add model/artifacts/bbl_model.pkl model/artifacts/bbl_model_meta.json
git add data/processed/bbl_matches_enriched.csv data/processed/bbl_features.csv
git commit -m "fix: latencia predictor + Paderborn mapping + calibración Platt"
git push
```

Y dejar este resumen de resultados para Claude:
- Latencia nueva: __ ms
- Cobertura 2023-24 nueva: __%  (era 89.6%)
- AUC final BBL: __ (era 0.660)
- Deploy actualizado: sí/no

---

## Estructura del proyecto

```
Predicción_Baloncesto/
├── api/
│   └── main.py                  # FastAPI endpoints
├── frontend/
│   ├── index.html
│   ├── app.js                   # UI (selector ACB/BBL)
│   └── styles.css
├── model/
│   ├── artifacts/
│   │   ├── bbl_model.pkl        # Modelo BBL v3 (XGBoost, 46 features)
│   │   ├── bbl_model_meta.json  # AUC, feature_cols, temporadas
│   │   ├── model.pkl            # Modelo ACB
│   │   └── model_meta.json
│   ├── bbl_feature_engineering.py
│   ├── bbl_predictor.py         # ← TAREA 1 aquí
│   ├── bbl_train.py             # ← TAREA 3/4 aquí
│   └── predictor.py             # Predictor ACB
├── scraper/
│   ├── bbl_merge_boxscores.py   # ← TAREA 2 aquí
│   ├── flashscore_boxscore_scraper.py
│   └── flashscore_index_scraper.py
├── data/
│   ├── processed/
│   │   ├── bbl_matches_enriched.csv  # 1,630 partidos + boxscores
│   │   ├── bbl_features.csv          # 1,491 filas, 46 features
│   │   ├── bbl_matches.csv           # CSV base BBL
│   │   └── matches.csv               # CSV base ACB
│   └── flashscore/
│       ├── index/                    # JSON índices por temporada
│       └── boxscores/                # JSONL boxscores por temporada
├── Procfile                     # Railway: uvicorn api.main:app
└── requirements.txt
```

---

## Comandos útiles

```bash
# Correr API local
uvicorn api.main:app --reload --port 8000

# Re-entrenar BBL completo
python3 -m scraper.bbl_merge_boxscores
python3 -m model.bbl_feature_engineering
python3 -m model.bbl_train

# Test predictor directo
python3 -c "from model.bbl_predictor import predict; print(predict('ALBA Berlin','FC Bayern Muenchen Basketball'))"

# Deploy a Railway
railway up --service perpetual-encouragement
```
