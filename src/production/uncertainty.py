"""Empirical residual-based prediction intervals.

IMPORTANT: these are *empirical residual-based prediction intervals*, NOT
calibrated probabilistic confidence intervals. Ridge is a point estimator; the
interval says "in the temporal holdout, the model's relative errors fell inside
this band about X% of the time". Coverage on future data is not guaranteed.

Method: residuals are computed on the log1p scale (the scale the model is trained
on), r = log1p(actual) - log1p(predicted). Lower/upper quantiles of r are applied
multiplicatively, so the band widens for expensive long-haul loads and narrows for
short ones. The residuals come from the temporal-holdout model (trained Jan-Aug,
scored Sep-Oct), never from rows the scored model was fitted on.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DISCLAIMER = (
    "These are empirical residual-based prediction intervals, not calibrated "
    "probabilistic confidence intervals."
)
METHOD = "empirical_log_residual_quantiles"


def log_residuals(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1.0)
    return np.log1p(y_true) - np.log1p(y_pred)


def _quantiles(residuals: np.ndarray, coverage: float) -> tuple[float, float]:
    tail = (1.0 - coverage) / 2.0
    lo, hi = np.quantile(residuals, [tail, 1.0 - tail])
    return float(lo), float(hi)


def _coverage(residuals: np.ndarray, lo: float, hi: float) -> float:
    return float(np.mean((residuals >= lo) & (residuals <= hi)))


def compute_interval_metadata(
    y_true,
    y_pred,
    *,
    dates: Optional[pd.Series] = None,
    coverage: float = 0.90,
    source: str = "temporal holdout residuals",
) -> dict:
    """Build the interval metadata stored with a model version.

    If ``dates`` are supplied the holdout is split in time: quantiles are fitted on
    the earlier half and coverage is *checked* on the later half, so the reported
    coverage is out-of-sample rather than self-fulfilling.
    """
    if not 0.5 <= coverage < 1.0:
        raise ValueError("coverage must be in [0.5, 1.0)")
    res = log_residuals(y_true, y_pred)
    if len(res) < 50:
        raise ValueError("need at least 50 residuals to estimate an interval")
    lo, hi = _quantiles(res, coverage)

    meta = {
        "method": METHOD,
        "scale": "log1p",
        "nominal_coverage": coverage,
        "lower_quantile": lo,
        "upper_quantile": hi,
        "n_residuals": int(len(res)),
        "calibration_source": source,
        "in_sample_coverage": _coverage(res, lo, hi),
        "holdout_check": None,
        "disclaimer": DISCLAIMER,
    }

    if dates is not None:
        d = pd.to_datetime(pd.Series(np.asarray(dates)))
        cut = d.sort_values().iloc[len(d) // 2]
        early, late = (d < cut).to_numpy(), (d >= cut).to_numpy()
        if early.sum() >= 50 and late.sum() >= 50:
            elo, ehi = _quantiles(res[early], coverage)
            meta["holdout_check"] = {
                "calibrated_on": f"{d[early].min().date()} to {d[early].max().date()}",
                "evaluated_on": f"{d[late].min().date()} to {d[late].max().date()}",
                "n_calibration": int(early.sum()),
                "n_evaluation": int(late.sum()),
                "empirical_coverage": _coverage(res[late], elo, ehi),
                "nominal_coverage": coverage,
            }
    return meta


def apply_interval(predicted_rate, interval: Optional[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) rate arrays for point predictions.

    Guarantees lower <= predicted <= upper and lower >= 0.
    """
    pred = np.maximum(np.asarray(predicted_rate, dtype=float), 1.0)
    if not interval:
        return pred.copy(), pred.copy()
    log_pred = np.log1p(pred)
    lower = np.expm1(log_pred + float(interval["lower_quantile"]))
    upper = np.expm1(log_pred + float(interval["upper_quantile"]))
    return np.maximum(np.minimum(lower, pred), 0.0), np.maximum(upper, pred)
