"""Model performance monitoring once actual target values arrive.

Sign convention: bias (mean error) = mean(predicted - actual); positive means the
model over-predicts. MAPE ignores rows whose actual rate is <= 0.

If no actual rates have been supplied we say so; we never invent accuracy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

UNAVAILABLE_MESSAGE = "actual target data unavailable"
DISTANCE_BINS = [0, 200, 500, 1000, 1500, 2000, np.inf]
DISTANCE_LABELS = ["<=200", "201-500", "501-1000", "1001-1500", "1501-2000", ">2000"]
MIN_ROWS = 30
WARNING_RATIO = 1.10   # current MAE / baseline MAE above this -> "warning"
DEGRADED_RATIO = 1.25  # ... above this -> "degraded"


def compute_metrics(actual, predicted) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    err = p - a
    pos = a > 0
    return {
        "n": int(len(a)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_percent": float(np.mean(np.abs(err[pos]) / a[pos]) * 100.0) if pos.any() else None,
        "bias_mean_error": float(np.mean(err)),
    }


def _group_metrics(df: pd.DataFrame, key: pd.Series, min_rows: int = 1) -> list[dict]:
    rows = []
    for name, part in df.groupby(key, observed=True):
        if len(part) >= min_rows:
            rows.append({"group": str(name), **compute_metrics(part["actual_rate"], part["predicted_rate"])})
    return rows


def join_predictions_with_actuals(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Join on load_id (last prediction per load wins). ``actuals`` needs load_id, actual_rate."""
    if "load_id" not in actuals.columns or "actual_rate" not in actuals.columns:
        raise ValueError("actuals must contain columns: load_id, actual_rate")
    preds = predictions.drop_duplicates("load_id", keep="last")
    actual = actuals.drop_duplicates("load_id", keep="last")[["load_id", "actual_rate"]]
    return preds.merge(actual, on="load_id", how="inner")


def evaluate_performance(
    data: Union[pd.DataFrame, list, None],
    baseline: Optional[dict] = None,
    *,
    min_rows: int = MIN_ROWS,
) -> dict:
    """``data``: DataFrame/list of dicts with load_id, predicted_rate, actual_rate and
    optionally equipment, distance, date (used for the group breakdowns)."""
    frame = pd.DataFrame(data) if data is not None and not isinstance(data, pd.DataFrame) else data
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if frame is None or len(frame) == 0 or "actual_rate" not in frame.columns:
        return {"available": False, "message": UNAVAILABLE_MESSAGE, "generated_at": generated}

    frame = frame.copy()
    for col in ("predicted_rate", "actual_rate"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[np.isfinite(frame["predicted_rate"]) & np.isfinite(frame["actual_rate"])]
    if frame.empty:
        return {"available": False, "message": UNAVAILABLE_MESSAGE, "generated_at": generated}

    report: dict[str, Any] = {
        "available": True,
        "generated_at": generated,
        "overall": compute_metrics(frame["actual_rate"], frame["predicted_rate"]),
        "by_equipment": [],
        "by_distance_bucket": [],
        "by_month": [],
    }
    if "equipment" in frame.columns:
        report["by_equipment"] = _group_metrics(frame, frame["equipment"])
    if "distance" in frame.columns:
        bucket = pd.cut(pd.to_numeric(frame["distance"], errors="coerce"), DISTANCE_BINS, labels=DISTANCE_LABELS)
        report["by_distance_bucket"] = _group_metrics(frame, bucket)
    if "date" in frame.columns:
        month = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m")
        report["by_month"] = _group_metrics(frame, month)

    report["baseline_comparison"] = compare_to_baseline(report["overall"], baseline, min_rows)
    return report


def compare_to_baseline(current: dict, baseline: Optional[dict], min_rows: int = MIN_ROWS) -> dict:
    if not baseline or baseline.get("MAE") in (None, 0):
        return {"status": "no_baseline", "message": "no baseline metrics available"}
    base_mae = float(baseline["MAE"])
    ratio = current["MAE"] / base_mae
    out = {
        "baseline_source": "temporal holdout (model metadata)",
        "baseline_MAE": base_mae,
        "current_MAE": current["MAE"],
        "mae_ratio": ratio,
        "mae_change_percent": (ratio - 1.0) * 100.0,
        "baseline_RMSE": baseline.get("RMSE"),
        "current_RMSE": current["RMSE"],
        "thresholds": {"warning_ratio": WARNING_RATIO, "degraded_ratio": DEGRADED_RATIO, "min_rows": min_rows},
    }
    if current["n"] < min_rows:
        out["status"] = "insufficient_data"
        out["message"] = f"only {current['n']} matched rows (< {min_rows}); comparison not reliable"
    elif ratio > DEGRADED_RATIO:
        out["status"] = "degraded"
    elif ratio > WARNING_RATIO:
        out["status"] = "warning"
    else:
        out["status"] = "ok"
    return out
