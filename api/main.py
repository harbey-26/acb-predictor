"""
FastAPI backend para el predictor de baloncesto (ACB + BBL).

Endpoints:
  GET  /teams?liga=acb     → lista de equipos disponibles
  POST /predict            → predicción (campo 'liga' en el body: 'acb'|'bbl')
  GET  /health             → estado del servicio
  GET  /leagues            → ligas disponibles

Inicio:
  uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import PredictRequest, PredictResponse, TeamInfo
from model.predictor import predict as acb_predict, get_available_teams as acb_teams
import model.bbl_predictor as bbl

app = FastAPI(
    title="Basketball Predictor API",
    description="Predicción de partidos: Liga ACB española y Basketball Bundesliga alemana",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    if bbl.is_available():
        bbl_status = "XGBoost v3 + boxscores FlashScore (AUC 0.660)"
    else:
        bbl_status = "not_trained"
    return {
        "status": "ok",
        "acb_model": "XGBoost v2 + Platt calibration (69.9%)",
        "bbl_model": bbl_status,
    }


@app.get("/leagues")
def leagues():
    """Devuelve las ligas disponibles."""
    result = [{"id": "acb", "nombre": "Liga ACB (España)", "disponible": True}]
    result.append({
        "id": "bbl",
        "nombre": "Basketball Bundesliga (Alemania)",
        "disponible": bbl.is_available(),
    })
    return result


@app.get("/teams", response_model=List[TeamInfo])
def teams(response: Response, liga: str = Query("acb", description="Liga: 'acb' o 'bbl'")):
    """Devuelve la lista de equipos disponibles para la liga seleccionada."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    liga = liga.lower()
    if liga == "bbl":
        if not bbl.is_available():
            raise HTTPException(status_code=503, detail="Modelo BBL no disponible aún. Los datos están siendo recopilados.")
        return bbl.get_available_teams()
    return acb_teams()


@app.post("/predict", response_model=PredictResponse)
def predict_match(req: PredictRequest):
    """
    Predice el resultado de un partido.
    - **liga**: 'acb' (default) o 'bbl'
    - **equipo_local**: nombre del equipo local
    - **equipo_visitante**: nombre del equipo visitante
    """
    liga = (req.liga or "acb").lower()
    try:
        if liga == "bbl":
            if not bbl.is_available():
                raise HTTPException(status_code=503, detail="Modelo BBL no disponible. Los datos están siendo procesados.")
            result = bbl.predict(req.equipo_local, req.equipo_visitante)
        else:
            result = acb_predict(req.equipo_local, req.equipo_visitante)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
