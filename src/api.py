"""
FastAPI serving layer for the Zomato delivery-time model (Phase 2).

Endpoints:
  GET  /health      — liveness/readiness (K8s probes). Confirms artifacts load.
  GET  /model_info  — current model MAPE/MAE/R2 + training date + version tag.
  POST /predict     — batch prediction with the full business layer per record.

MODEL_VERSION is the tag the canary rollout keys off in prediction logs/metrics.
"""
from __future__ import annotations

import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import predict as P

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")
app = FastAPI(title="Zomato Delivery-Time API", version=MODEL_VERSION)


class DeliveryRecord(BaseModel):
    Delivery_person_Age: float | None = None
    Delivery_person_Ratings: float | None = None
    Restaurant_latitude: float | None = None
    Restaurant_longitude: float | None = None
    Delivery_location_latitude: float | None = None
    Delivery_location_longitude: float | None = None
    Order_Date: str | None = None
    Time_Orderd: str | None = None
    Time_Order_picked: str | None = None
    Weather_conditions: str | None = None
    Road_traffic_density: str | None = None
    Vehicle_condition: int | None = None
    Type_of_order: str | None = None
    Type_of_vehicle: str | None = None
    multiple_deliveries: float | None = None
    Festival: str | None = None
    City: str | None = None
    distance_km: float | None = None


class PredictRequest(BaseModel):
    records: list[DeliveryRecord] = Field(..., min_length=1)


@app.get("/health")
def health():
    try:
        P._load_artifacts()
        return {"status": "ok", "model_version": MODEL_VERSION}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"artifacts unavailable: {e}")


@app.get("/model_info")
def model_info():
    try:
        return {"model_version": MODEL_VERSION, **P.get_model_info()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        records = [r.model_dump() for r in req.records]
        results = P.predict_records(records)
        return {"model_version": MODEL_VERSION, "results": results}
    except Exception as e:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=400, detail=str(e))
