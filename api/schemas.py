"""Pydantic request/response models (OpenAPI docs are generated from these)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from production.validation import ALLOWED_EQUIPMENT

EXAMPLE_REQUEST = {
    "load_id": "LOAD_001",
    "date": "2025-11-15",
    "distance": 850,
    "weight": 32000,
    "pickup_lat": 40.7128,
    "pickup_lon": -74.0060,
    "delivery_lat": 41.8781,
    "delivery_lon": -87.6298,
    "equipment": "Dry Van",
    "market_index": 1.05,
    "quote_signal": 2.0,
}


class PredictionRequest(BaseModel):
    """Raw load features. Engineered features are computed server-side."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    load_id: str = Field(..., min_length=1, max_length=64, description="Caller's identifier for the load.")
    date: dt.date = Field(..., description="Load date (YYYY-MM-DD).")
    distance: float = Field(..., gt=0, allow_inf_nan=False, description="Road distance in miles (> 0).")
    weight: Optional[float] = Field(None, allow_inf_nan=False, description="Weight in lb. May be omitted; negative values are flagged, not rejected.")
    pickup_lat: float = Field(..., ge=-90, le=90, allow_inf_nan=False)
    pickup_lon: float = Field(..., ge=-180, le=180, allow_inf_nan=False)
    delivery_lat: float = Field(..., ge=-90, le=90, allow_inf_nan=False)
    delivery_lon: float = Field(..., ge=-180, le=180, allow_inf_nan=False)
    equipment: str = Field(..., description=f"One of {list(ALLOWED_EQUIPMENT)} (case-sensitive).")
    market_index: Optional[float] = Field(None, allow_inf_nan=False, description="May be omitted; a training fallback is used.")
    quote_signal: Optional[float] = Field(None, allow_inf_nan=False, description="May be omitted; the training median is used.")
    pickup: Optional[str] = Field(None, max_length=100, description="City name. Accepted but ignored by the model.")
    delivery: Optional[str] = Field(None, max_length=100, description="City name. Accepted but ignored by the model.")

    @field_validator("load_id")
    @classmethod
    def _load_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("load_id must not be blank")
        return value

    @field_validator("equipment")
    @classmethod
    def _equipment(cls, value: str) -> str:
        if value not in ALLOWED_EQUIPMENT:
            raise ValueError(f"unsupported equipment '{value[:40]}'; allowed: {list(ALLOWED_EQUIPMENT)}")
        return value

    def to_record(self) -> dict[str, Any]:
        record = self.model_dump(exclude={"pickup", "delivery"})
        record["date"] = self.date.isoformat()
        return record


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"examples": [{
        "load_id": "LOAD_001", "predicted_rate": 1850.25, "lower_rate": 1500.10, "upper_rate": 2200.80,
        "interval_coverage": 0.9, "interval_method": "empirical_log_residual_quantiles",
        "model_version": "v1", "notes": [],
    }]})

    load_id: str
    predicted_rate: float = Field(..., description="Point estimate in dollars.")
    lower_rate: float = Field(..., description="Lower bound of the empirical residual-based interval.")
    upper_rate: float = Field(..., description="Upper bound of the empirical residual-based interval.")
    interval_coverage: Optional[float] = Field(None, description="NOMINAL coverage in the temporal holdout; not a calibrated guarantee.")
    interval_method: Optional[str] = Field(None, description="Empirical residual-based prediction interval, not a calibrated probabilistic confidence interval.")
    model_version: str
    notes: list[str] = Field(default_factory=list, description="How missing/odd optional inputs were handled.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model: str
    version: str
    alpha: Optional[float] = None
    feature_count: Optional[int] = None
    target_transform: Optional[str] = None
    training_rows: Optional[int] = None
    training_date: Optional[str] = None
    training_data_period: Optional[dict] = None
    metrics: Optional[dict] = None
    prediction_interval: Optional[dict] = None

    model_config = ConfigDict(protected_namespaces=())


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[list] = None
