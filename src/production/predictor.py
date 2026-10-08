"""Raw input -> validation -> cleaning -> features -> model -> inverse transform.

Reuses the assessment ``apply_cleaning`` / ``build_features`` unchanged, and applies
the same post-processing as ``src/predict_validation.py`` (expm1, floor at 1.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import production  # noqa: F401  (adds src/ to sys.path)
from data_cleaning import apply_cleaning
from features import build_features, get_model_features
from production.logging_config import get_logger
from production.model_loader import LoadedModel, load_artifact_file
from production.uncertainty import apply_interval
from production.validation import (
    SMOKE_RECORD, InputValidationError, normalize_frame, validate_dataframe,
)

log = get_logger("predictor")


class ModelFeatureError(RuntimeError):
    """The model expects features the current feature pipeline cannot produce."""


def expected_feature_count() -> int:
    """Feature count produced by the *current* cleaning + feature code."""
    from data_cleaning import CleaningStatistics

    stats = CleaningStatistics(weight_median=0.0, market_index_global_median=1.0, market_index_by_date={})
    frame = normalize_frame(pd.DataFrame([SMOKE_RECORD]))
    return len(get_model_features(build_features(apply_cleaning(frame, stats))))


class Predictor:
    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded

    @property
    def version(self) -> str:
        return self.loaded.version

    @property
    def interval_metadata(self) -> Optional[dict]:
        return self.loaded.metadata.get("prediction_interval")

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return load_id, predicted_rate (unrounded), lower_rate, upper_rate."""
        report = validate_dataframe(df)
        if not report.valid:
            raise InputValidationError(report)

        frame = normalize_frame(df)
        features = build_features(apply_cleaning(frame, self.loaded.cleaning_stats))
        missing = [c for c in self.loaded.feature_columns if c not in features.columns]
        if missing:
            raise ModelFeatureError(f"missing model features: {missing}")

        log_pred = self.loaded.model.predict(features[self.loaded.feature_columns])
        predicted = np.maximum(np.expm1(log_pred), 1.0)
        lower, upper = apply_interval(predicted, self.interval_metadata)
        return pd.DataFrame({
            "load_id": frame["load_id"].to_numpy(),
            "predicted_rate": predicted,
            "lower_rate": lower,
            "upper_rate": upper,
        })

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = self.predict_frame(pd.DataFrame(records))
        return out.to_dict(orient="records")


def record_notes(record: dict[str, Any]) -> list[str]:
    """Human-readable notes about how optional/odd inputs were handled."""
    def missing(key: str) -> bool:
        v = record.get(key)
        return v is None or (isinstance(v, float) and np.isnan(v))

    notes = []
    if missing("weight"):
        notes.append("weight missing: training median used and missing-weight flag set")
    elif record.get("weight") is not None and float(record["weight"]) < 0:
        notes.append("weight is negative: kept as raw value and flagged by the cleaning layer")
    if missing("market_index"):
        notes.append("market_index missing: date-level or global training fallback used")
    if missing("quote_signal"):
        notes.append("quote_signal missing: training median imputed by the model pipeline")
    return notes


def smoke_test_artifact(model_file: Path, metadata: dict) -> list[str]:
    """Load an artifact and predict one record end-to-end. Returns problems (empty = ok)."""
    problems = []
    artifact = load_artifact_file(model_file)
    columns = list(artifact["feature_columns"])
    if len(columns) != int(metadata["feature_count"]):
        problems.append(f"artifact has {len(columns)} features but metadata says {metadata['feature_count']}")
    if columns != list(metadata["feature_names"]):
        problems.append("artifact feature order differs from metadata feature_names")
    loaded = LoadedModel(str(metadata.get("version")), artifact["model"], columns, artifact["cleaning_stats"], metadata)
    try:
        result = Predictor(loaded).predict_frame(pd.DataFrame([SMOKE_RECORD]))
        value = float(result["predicted_rate"].iloc[0])
        if not np.isfinite(value) or value <= 0:
            problems.append(f"smoke prediction not finite/positive: {value}")
    except Exception as exc:
        problems.append(f"smoke prediction failed: {type(exc).__name__}: {exc}")
    return problems
