"""
FastAPI backend para el predictor de la Liga ACB.

Endpoints:
  GET  /teams           → lista de equipos disponibles
  POST /predict         → predicción de un partido
  GET  /health          → estado del servicio

Inicio:
  uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import PredictRequest, PredictResponse, TeamInfo
from model.predictor import predict, get_available_teams

app = FastAPI(
    title="ACB Predictor API",
    description="Predicción de partidos de la Liga ACB española",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir el frontend estático
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost + Platt calibration", "accuracy_test": "69.3%"}


@app.get("/teams", response_model=list[TeamInfo])
def teams():
    """Devuelve la lista de equipos disponibles para predecir."""
    return get_available_teams()


@app.post("/predict", response_model=PredictResponse)
def predict_match(req: PredictRequest):
    """
    Predice el resultado de un partido.

    - **equipo_local**: nombre (parcial) del equipo que juega en casa
    - **equipo_visitante**: nombre (parcial) del equipo visitante
    """
    try:
        result = predict(req.equipo_local, req.equipo_visitante)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
