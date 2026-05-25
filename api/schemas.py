from pydantic import BaseModel
from typing import Dict, List, Optional


class TeamInfo(BaseModel):
    club_id: int
    nombre: str


class PredictRequest(BaseModel):
    equipo_local: str
    equipo_visitante: str


class FeaturesSummary(BaseModel):
    home_win_rate_reciente: float
    away_win_rate_reciente: float
    home_pts_promedio: float
    away_pts_promedio: float
    home_val_promedio: float
    away_val_promedio: float
    h2h_enfrentamientos: int
    h2h_tasa_local: float


class PredictResponse(BaseModel):
    equipo_local: str
    equipo_visitante: str
    prob_local: float
    prob_visitante: float
    prediccion: str        # "local" | "visitante"
    confianza: str         # "alta" | "media" | "baja"
    features_usadas: FeaturesSummary
