from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List

from predict import predict_dam_from_payload

app = FastAPI(title="DAM Forecast API", version="1.0")


class DAMPredictionRequest(BaseModel):
    prediction_date: str
    region: str = "Telangana"
    market_data: List[Dict[str, Any]]
    weather_data: List[Dict[str, Any]]


@app.get("/")
def root():
    return {"status": "ok", "service": "DAM Forecast API"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "DAM Forecast API"}


@app.post("/predict")
def predict(payload: DAMPredictionRequest):
    try:
        return predict_dam_from_payload(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))