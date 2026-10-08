"""Prediction monitoring: structured JSONL logging + summary statistics.

One JSON object per line in ``monitoring/predictions/predictions.jsonl``:
    event="prediction": timestamp, load_id, predicted_rate, lower_rate, upper_rate,
                        model_version, latency_ms, inputs{...drift features...}
    event="error":      timestamp, error_type, status_code, model_version, latency_ms
Error records never contain request payloads or exception messages.

Privacy: no secrets, no free text, no client identifiers. ``inputs`` holds only the
freight features used for drift detection and can be disabled (LOG_INPUT_FEATURES=false).
Logging failures never break a prediction.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from production.logging_config import get_logger

log = get_logger("monitoring")

NO_DATA_MESSAGE = "No production prediction data available."
LOGGED_INPUTS = (
    "distance", "weight", "market_index", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "equipment", "date",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value if isinstance(value, (str, int, bool)) else str(value)


class PredictionMonitor:
    def __init__(self, path: Path, *, enabled: bool = True, log_inputs: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.log_inputs = log_inputs
        self._lock = threading.Lock()

    def _append(self, record: dict) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            return True
        except OSError as exc:
            log.warning("could not write monitoring record", extra={"event": "monitoring_write_failed", "error": type(exc).__name__})
            return False

    def log_prediction(self, *, load_id: str, predicted_rate: float, model_version: str, latency_ms: float,
                       lower_rate: Optional[float] = None, upper_rate: Optional[float] = None,
                       inputs: Optional[dict] = None) -> bool:
        record = {
            "timestamp": _now(), "event": "prediction", "load_id": load_id,
            "predicted_rate": _clean(predicted_rate), "lower_rate": _clean(lower_rate), "upper_rate": _clean(upper_rate),
            "model_version": model_version, "latency_ms": round(float(latency_ms), 3),
        }
        if self.log_inputs and inputs:
            record["inputs"] = {k: _clean(inputs.get(k)) for k in LOGGED_INPUTS if k in inputs}
        return self._append(record)

    def log_error(self, *, error_type: str, status_code: int, latency_ms: float, model_version: Optional[str] = None) -> bool:
        return self._append({
            "timestamp": _now(), "event": "error", "error_type": error_type, "status_code": int(status_code),
            "model_version": model_version, "latency_ms": round(float(latency_ms), 3),
        })


def read_records(path: Path) -> tuple[list[dict], int]:
    """Return (records, malformed_line_count). Missing file -> ([], 0)."""
    path = Path(path)
    if not path.is_file():
        return [], 0
    records, bad = [], 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj) if isinstance(obj, dict) else None
                bad += 0 if isinstance(obj, dict) else 1
            except json.JSONDecodeError:
                bad += 1
    return records, bad


def records_to_prediction_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten prediction events (inputs expanded into columns) for drift/performance."""
    rows = []
    for r in records:
        if r.get("event") != "prediction":
            continue
        rows.append({"load_id": r.get("load_id"), "predicted_rate": r.get("predicted_rate"),
                     "model_version": r.get("model_version"), "timestamp": r.get("timestamp"),
                     **(r.get("inputs") or {})})
    return pd.DataFrame(rows)


def summarize_predictions(path: Path) -> dict:
    records, malformed = read_records(path)
    preds = [r for r in records if r.get("event") == "prediction" and r.get("predicted_rate") is not None]
    errors = [r for r in records if r.get("event") == "error"]
    if not preds and not errors:
        return {"available": False, "message": NO_DATA_MESSAGE, "malformed_lines": malformed}

    summary: dict[str, Any] = {
        "available": True, "n_predictions": len(preds), "error_count": len(errors),
        "malformed_lines": malformed,
        "errors_by_type": pd.Series([e.get("error_type") for e in errors]).value_counts().to_dict() if errors else {},
    }
    if preds:
        values = np.array([p["predicted_rate"] for p in preds], dtype=float)
        lat = np.array([p["latency_ms"] for p in preds if p.get("latency_ms") is not None], dtype=float)
        counts, edges = np.histogram(values, bins=10)
        summary.update({
            "mean_prediction": float(values.mean()),
            "median_prediction": float(np.median(values)),
            "min_prediction": float(values.min()),
            "max_prediction": float(values.max()),
            "prediction_quantiles": {f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 90)},
            "prediction_distribution": {"bin_edges": [float(e) for e in edges], "counts": counts.tolist()},
            "avg_latency_ms": float(lat.mean()) if len(lat) else None,
            "p95_latency_ms": float(np.percentile(lat, 95)) if len(lat) else None,
            "by_model_version": pd.Series([p.get("model_version") for p in preds]).value_counts().to_dict(),
            "first_timestamp": min(p["timestamp"] for p in preds),
            "last_timestamp": max(p["timestamp"] for p in preds),
        })
    return summary
