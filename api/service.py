"""Framework-independent service object: model lifecycle + one prediction request.

Kept free of FastAPI imports so the request logic can be tested directly.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from production.config import Settings, get_settings
from production.logging_config import get_logger
from production.model_loader import ModelNotAvailableError, load_production_model
from production.monitoring import PredictionMonitor
from production.predictor import Predictor, record_notes
from production.validation import InputValidationError

log = get_logger("service")


class ModelService:
    def __init__(self, settings: Optional[Settings] = None, predictor: Optional[Predictor] = None):
        self.settings = settings or get_settings()
        self.predictor = predictor
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()
        self.monitor = PredictionMonitor(
            self.settings.prediction_log_path,
            enabled=self.settings.log_predictions,
            log_inputs=self.settings.log_input_features,
        )

    # ---- model lifecycle ---------------------------------------------------
    def ensure_loaded(self) -> bool:
        """Load the promoted model if not loaded yet. Never trains. Never raises."""
        if self.predictor is not None:
            return True
        with self._lock:
            if self.predictor is not None:
                return True
            try:
                self.predictor = Predictor(load_production_model(self.settings))
                self.last_error = None
            except ModelNotAvailableError as exc:
                self.last_error = str(exc)
                log.error("production model not available", extra={"event": "model_unavailable", "reason": self.last_error})
            return self.predictor is not None

    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None

    def get_predictor(self) -> Predictor:
        if not self.ensure_loaded():
            raise ModelNotAvailableError(self.last_error or "model not loaded")
        return self.predictor  # type: ignore[return-value]

    def model_info(self) -> dict[str, Any]:
        meta = self.get_predictor().loaded.metadata
        interval = meta.get("prediction_interval") or {}
        return {
            "model": meta.get("algorithm", "Ridge"),
            "version": self.get_predictor().version,
            "alpha": meta.get("alpha"),
            "feature_count": meta.get("feature_count"),
            "target_transform": meta.get("target_transformation"),
            "training_rows": meta.get("training_rows"),
            "training_date": meta.get("training_date"),
            "training_data_period": meta.get("training_data_period"),
            "metrics": {k: v for k, v in (meta.get("metrics") or {}).items() if isinstance(v, (int, float))},
            "prediction_interval": {
                "method": interval.get("method"),
                "nominal_coverage": interval.get("nominal_coverage"),
                "holdout_check": interval.get("holdout_check"),
                "disclaimer": interval.get("disclaimer"),
            } if interval else None,
        }

    # ---- request handling ------------------------------------------------------
    def record_error(self, error_type: str, status_code: int, latency_ms: float = 0.0) -> None:
        version = self.predictor.version if self.predictor else None
        self.monitor.log_error(error_type=error_type, status_code=status_code, latency_ms=latency_ms, model_version=version)

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        """Predict one validated record. Raises ModelNotAvailableError / InputValidationError /
        other exceptions; each failure is logged and counted before being re-raised."""
        start = time.perf_counter()
        try:
            predictor = self.get_predictor()
            result = predictor.predict_records([record])[0]
        except ModelNotAvailableError:
            self.record_error("model_unavailable", 503, (time.perf_counter() - start) * 1000)
            raise
        except InputValidationError as exc:
            log.warning("input validation failed", extra={"event": "validation_failed", "errors": exc.report.errors})
            self.record_error("input_validation", 422, (time.perf_counter() - start) * 1000)
            raise
        except Exception:
            log.exception("prediction failed", extra={"event": "prediction_failed"})
            self.record_error("internal_error", 500, (time.perf_counter() - start) * 1000)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        interval = predictor.interval_metadata or {}
        response = {
            "load_id": str(result["load_id"]),
            "predicted_rate": round(float(result["predicted_rate"]), 2),
            "lower_rate": round(float(result["lower_rate"]), 2),
            "upper_rate": round(float(result["upper_rate"]), 2),
            "interval_coverage": interval.get("nominal_coverage"),
            "interval_method": interval.get("method"),
            "model_version": predictor.version,
            "notes": record_notes(record),
        }
        self.monitor.log_prediction(
            load_id=response["load_id"], predicted_rate=response["predicted_rate"],
            lower_rate=response["lower_rate"], upper_rate=response["upper_rate"],
            model_version=predictor.version, latency_ms=latency_ms, inputs=record,
        )
        log.info("prediction served", extra={
            "event": "prediction", "load_id": response["load_id"], "model_version": predictor.version,
            "latency_ms": round(latency_ms, 2),
        })
        return response
