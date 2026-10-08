from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    EXAMPLE_REQUEST, ErrorResponse, HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse,
)
from production.model_loader import ModelNotAvailableError
from production.validation import InputValidationError

router = APIRouter()

MODEL_UNAVAILABLE = "The production model is not available. Contact the service operator."


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}, tags=["ops"])
def health(request: Request):
    """Liveness + model readiness. Returns 503 while no model is loaded."""
    loaded = request.app.state.service.ensure_loaded()
    body = {"status": "healthy" if loaded else "unhealthy", "model_loaded": loaded}
    return JSONResponse(status_code=200 if loaded else 503, content=body)


@router.get("/model/info", response_model=ModelInfoResponse, responses={503: {"model": ErrorResponse}}, tags=["ops"])
def model_info(request: Request):
    """Metadata of the currently promoted model."""
    try:
        return request.app.state.service.model_info()
    except ModelNotAvailableError:
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(payload: PredictionRequest, request: Request):
    """Predict the posted rate for one load from RAW features.

    Pipeline: validation -> cleaning -> feature engineering -> Ridge (log1p target) -> expm1.
    `lower_rate`/`upper_rate` are **empirical residual-based prediction intervals from the
    temporal validation residuals, not calibrated probabilistic confidence intervals**.
    """
    service = request.app.state.service
    try:
        return service.predict_one(payload.to_record())
    except ModelNotAvailableError:
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.report.errors)
