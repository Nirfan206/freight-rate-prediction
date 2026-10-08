"""FastAPI application.

Run:  python -m uvicorn api.main:app --reload --port 8000     then open /docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import api  # noqa: F401  (puts src/ on sys.path)
from api.routes import router
from api.service import ModelService
from production.config import Settings, get_settings
from production.logging_config import configure_logging, get_logger

log = get_logger("api")

DESCRIPTION = """
Predicts freight **posted rate** from raw load features.

* The service **loads the promoted model from the registry** at startup; it never trains.
* `lower_rate` / `upper_rate` are *empirical residual-based prediction intervals* from the
  temporal validation residuals - **not** calibrated probabilistic confidence intervals.
* This is a local/deployable architecture; no public production deployment is claimed.
"""


def create_app(settings: Optional[Settings] = None, service: Optional[ModelService] = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    service = service or ModelService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("api starting", extra={"event": "api_startup", "registry": str(settings.registry_path)})
        if service.ensure_loaded():
            log.info("api ready", extra={"event": "api_ready", "model_version": service.predictor.version})
        else:
            log.error("api started WITHOUT a model; /predict will return 503", extra={"event": "api_started_without_model"})
        yield
        log.info("api shutting down", extra={"event": "api_shutdown"})

    app = FastAPI(
        title="Freight Rate Prediction API",
        version="1.0.0",
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.service = service
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        details = [
            {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "message": str(err["msg"])}
            for err in exc.errors()
        ]
        log.warning("request validation failed", extra={"event": "validation_failed", "fields": [d["field"] for d in details]})
        service.record_error("request_validation", 422)
        return JSONResponse(status_code=422, content={"error": "validation_error", "message": "Request validation failed", "details": details})

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException):
        if isinstance(exc.detail, list):  # e.g. validation errors raised by the /predict route
            body = {"error": "validation_error", "message": "Input validation failed", "details": exc.detail}
        else:
            body = {"error": "http_error", "message": str(exc.detail), "details": None}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        # Full traceback is logged server-side by the service; clients get a generic message.
        return JSONResponse(status_code=500, content={"error": "internal_error", "message": "Internal server error", "details": None})

    return app


app = create_app()
