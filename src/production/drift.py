"""Lightweight data-drift detection (no extra dependencies).

Numeric features: Population Stability Index (PSI) over quantile bins fixed from the
reference (training) data.  Categorical features (equipment): PSI over category
frequencies, with unseen categories pooled into "__other__"; the raw frequency
comparison is reported alongside.

Thresholds (conventional PSI rules of thumb):
    PSI < 0.10          -> "stable"
    0.10 <= PSI <= 0.25 -> "warning"
    PSI > 0.25          -> "drift"
A feature with fewer than MIN_SAMPLES current values is "insufficient_data".

Drift means the input distribution moved; it does NOT by itself prove the model is
wrong. Pair it with performance monitoring once actual rates arrive.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "distance", "weight", "market_index", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
)
CATEGORICAL_FEATURES = ("equipment",)
N_BINS = 10
MIN_SAMPLES = 30
EPSILON = 1e-4
PSI_WARNING = 0.10
PSI_DRIFT = 0.25
OTHER = "__other__"
STATUS_RANK = {"stable": 0, "warning": 1, "drift": 2}


def psi_status(value: float) -> str:
    if value < PSI_WARNING:
        return "stable"
    if value <= PSI_DRIFT:
        return "warning"
    return "drift"


def population_stability_index(reference_props, current_props) -> float:
    ref = np.clip(np.asarray(reference_props, dtype=float), EPSILON, None)
    cur = np.clip(np.asarray(current_props, dtype=float), EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / counts.sum()


def _clean_numeric(series: pd.Series) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()
    return v[np.isfinite(v)]


def build_reference_profile(df: pd.DataFrame, *, n_bins: int = N_BINS, source: str = "training data") -> dict:
    """Summarise the reference (training) distribution for later PSI comparisons."""
    features: dict[str, Any] = {}
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        values = _clean_numeric(df[col])
        if len(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1))[1:-1])
        features[col] = {
            "type": "numeric",
            "edges": [float(e) for e in edges],
            "proportions": _bin_proportions(values, edges).tolist(),
            "n": int(len(values)),
            "missing_rate": float(1.0 - len(values) / len(df)),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        freq = df[col].dropna().astype(str).value_counts(normalize=True)
        features[col] = {
            "type": "categorical",
            "frequencies": {str(k): float(v) for k, v in freq.items()},
            "n": int(freq.size and df[col].notna().sum()),
            "missing_rate": float(df[col].isna().mean()),
        }
    return {
        "source": source,
        "n_rows": int(len(df)),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": features,
    }


def _numeric_result(name: str, spec: dict, series: pd.Series, min_samples: int) -> dict:
    values = _clean_numeric(series)
    base = {"feature": name, "metric": "PSI", "n_current": int(len(values)),
            "current_missing_rate": float(1.0 - len(values) / max(len(series), 1)),
            "reference_missing_rate": spec.get("missing_rate")}
    if len(values) < min_samples:
        return {**base, "value": None, "status": "insufficient_data"}
    edges = np.asarray(spec["edges"], dtype=float)
    value = population_stability_index(spec["proportions"], _bin_proportions(values, edges))
    return {**base, "value": round(value, 6), "status": psi_status(value),
            "current_mean": float(values.mean()), "reference_mean": spec.get("mean")}


def _categorical_result(name: str, spec: dict, series: pd.Series, min_samples: int) -> dict:
    present = series.dropna().astype(str)
    base = {"feature": name, "metric": "PSI_categorical", "n_current": int(len(present))}
    if len(present) < min_samples:
        return {**base, "value": None, "status": "insufficient_data"}
    ref = spec["frequencies"]
    cur_freq = present.map(lambda x: x if x in ref else OTHER).value_counts(normalize=True)
    categories = sorted(set(ref) | set(cur_freq.index))
    ref_p = [ref.get(c, 0.0) for c in categories]
    cur_p = [float(cur_freq.get(c, 0.0)) for c in categories]
    value = population_stability_index(ref_p, cur_p)
    return {
        **base,
        "value": round(value, 6),
        "status": psi_status(value),
        "max_abs_frequency_diff": float(np.max(np.abs(np.asarray(cur_p) - np.asarray(ref_p)))),
        "frequencies": {c: {"reference": r, "current": c_} for c, r, c_ in zip(categories, ref_p, cur_p)},
    }


def detect_drift(current: pd.DataFrame, profile: dict, *, min_samples: int = MIN_SAMPLES, source: Optional[str] = None) -> dict:
    """Compare ``current`` raw inputs with a reference profile."""
    results = []
    for name, spec in profile["features"].items():
        if name not in current.columns:
            results.append({"feature": name, "metric": "PSI", "value": None, "status": "insufficient_data",
                            "note": "feature not present in current data"})
        elif spec["type"] == "numeric":
            results.append(_numeric_result(name, spec, current[name], min_samples))
        else:
            results.append(_categorical_result(name, spec, current[name], min_samples))

    ranked = [STATUS_RANK[r["status"]] for r in results if r["status"] in STATUS_RANK]
    overall = {v: k for k, v in STATUS_RANK.items()}[max(ranked)] if ranked else "insufficient_data"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "n_current_rows": int(len(current)),
        "reference": {"source": profile.get("source"), "n_rows": profile.get("n_rows")},
        "thresholds": {"stable_below": PSI_WARNING, "warning_up_to": PSI_DRIFT, "drift_above": PSI_DRIFT,
                       "min_samples": min_samples},
        "overall_status": overall,
        "features": results,
        "note": "Drift = input distribution shift versus training data; it does not prove the model is wrong.",
    }
