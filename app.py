import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_dam_from_payload

app = FastAPI(title="DAM Forecast API", version="7.0")


class DAMPredictionRequest(BaseModel):
    prediction_date: str = Field(..., description="Target forecast date in YYYY-MM-DD format")
    region: str = Field(default="Telangana", description="State or market region")
    market_data: List[Dict[str, Any]]
    weather_data: List[Dict[str, Any]]


@app.get("/")
def root():
    return {"status": "ok", "service": "DAM Forecast API", "version": "7.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "DAM Forecast API", "version": "7.0"}


@app.post("/predict")
def predict(payload: DAMPredictionRequest):
    try:
        return predict_dam_from_payload(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))