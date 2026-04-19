from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from final_lending.service import (
    get_available_user_ids,
    get_sales_forecast,
    get_summary,
    get_user_churn,
    get_user_recommendations,
)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    return value


app = FastAPI(
    title="Customer Experience API",
    version="1.0.0",
    description="FastAPI wrapper for churn scoring, retention recommendations, and sales forecast.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    try:
        get_summary()
        return {"ready": True}
    except Exception:
        return {"ready": False}


@app.get("/summary")
def summary() -> dict:
    return JSONResponse(content=_normalize(get_summary()))


@app.get("/users")
def users(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return JSONResponse(content=_normalize({"user_ids": get_available_user_ids(limit)}))


@app.get("/predict_churn/{user_id}")
def predict_churn(user_id: int) -> dict:
    try:
        return JSONResponse(content=_normalize(get_user_churn(user_id)))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = Query(default=5, ge=1, le=20)) -> dict:
    try:
        return JSONResponse(
            content=_normalize({"user_id": user_id, "recommendations": get_user_recommendations(user_id, top_n)})
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/forecast_sales")
def forecast_sales() -> dict:
    return JSONResponse(content=_normalize({"forecast": get_sales_forecast()}))
