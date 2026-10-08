"""Production input validation.

Design rule: reject *fundamentally invalid* records (impossible coordinates,
non-positive distance, unknown equipment, bad dates, non-numeric/non-finite
values, duplicate IDs). Merely *unusual or incomplete* values that the existing
assessment cleaning layer is built to handle (missing/negative weight, missing
market index, missing quote signal) are reported as warnings, never errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

ALLOWED_EQUIPMENT = ("Dry Van", "Reefer", "Flatbed")

REQUIRED_COLUMNS = (
    "load_id", "date", "distance", "equipment",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
)
OPTIONAL_COLUMNS = ("weight", "market_index", "quote_signal")
NUMERIC_COLUMNS = (
    "distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "weight", "market_index", "quote_signal",
)
COORDINATE_RANGES = {
    "pickup_lat": (-90.0, 90.0), "delivery_lat": (-90.0, 90.0),
    "pickup_lon": (-180.0, 180.0), "delivery_lon": (-180.0, 180.0),
}
# A missing-rate above this is flagged as "high" in the warning text (still a warning).
HIGH_MISSING_RATE = 0.20
WEIGHT_CAP = 47_500.0  # mirrors data_cleaning.WEIGHT_CAP

SMOKE_RECORD = {
    "load_id": "SMOKE_001", "date": "2025-11-15", "distance": 850.0, "weight": 32000.0,
    "pickup_lat": 40.7128, "pickup_lon": -74.0060, "delivery_lat": 41.8781, "delivery_lon": -87.6298,
    "equipment": "Dry Van", "market_index": 1.05, "quote_signal": 2.0,
}


@dataclass
class ValidationReport:
    valid: bool = True
    rows: int = 0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "rows": self.rows, "errors": list(self.errors), "warnings": list(self.warnings)}


class InputValidationError(ValueError):
    """Raised when input data fails production validation."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("; ".join(report.errors) or "invalid input")


def _pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%"


def _numeric(df: pd.DataFrame, column: str, report: ValidationReport) -> pd.Series:
    """Coerce to float; record errors for non-numeric and non-finite entries."""
    raw = df[column]
    values = pd.to_numeric(raw, errors="coerce").astype("float64")
    non_numeric = int((raw.notna() & values.isna()).sum())
    if non_numeric:
        report.errors.append(f"{column}: {non_numeric} non-numeric value(s)")
    non_finite = int(np.isinf(values).sum())
    if non_finite:
        report.errors.append(f"{column}: {non_finite} non-finite value(s)")
        values = values.replace([np.inf, -np.inf], np.nan)
    return values


def validate_dataframe(df: pd.DataFrame, *, require_target: bool = False) -> ValidationReport:
    """Validate a batch of raw records and return a structured report."""
    report = ValidationReport(rows=int(len(df)))
    n = len(df)
    if n == 0:
        report.errors.append("no rows supplied")
        report.valid = False
        return report

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.errors.append(f"missing required column(s): {', '.join(missing_cols)}")
    if require_target and "posted_rate" not in df.columns:
        report.errors.append("missing required column: posted_rate")

    # --- load_id -------------------------------------------------------
    if "load_id" in df.columns:
        ids = df["load_id"]
        blank = int((ids.isna() | (ids.astype(str).str.strip() == "")).sum())
        if blank:
            report.errors.append(f"load_id: {blank} missing/blank value(s)")
        dup_mask = ids.duplicated(keep=False) & ids.notna()
        if dup_mask.any():
            examples = ids[dup_mask].astype(str).unique()[:3].tolist()
            report.errors.append(f"load_id: {int(ids.duplicated().sum())} duplicate value(s), e.g. {examples}")

    # --- numeric columns --------------------------------------------------
    numeric: dict[str, pd.Series] = {}
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            numeric[column] = _numeric(df, column, report)

    if "distance" in numeric:
        d = numeric["distance"]
        null = int(d.isna().sum())
        if null:
            report.errors.append(f"distance: {null} missing value(s)")
        bad = int((d <= 0).sum())
        if bad:
            report.errors.append(f"distance: {bad} non-positive value(s) (must be > 0)")

    for column, (low, high) in COORDINATE_RANGES.items():
        if column in numeric:
            v = numeric[column]
            null = int(v.isna().sum())
            if null:
                report.errors.append(f"{column}: {null} missing value(s)")
            out = int(((v < low) | (v > high)).sum())
            if out:
                report.errors.append(f"{column}: {out} value(s) outside [{low:g}, {high:g}]")

    # --- date -------------------------------------------------------------------
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            report.errors.append(f"date: {bad} missing/invalid value(s)")

    # --- equipment ---------------------------------------------------------------
    if "equipment" in df.columns:
        eq = df["equipment"]
        unsupported = eq[~eq.isin(ALLOWED_EQUIPMENT)]
        if len(unsupported):
            shown = sorted({str(x) for x in unsupported.tolist()})[:5]
            report.errors.append(
                f"equipment: {len(unsupported)} unsupported value(s) {shown}; allowed: {list(ALLOWED_EQUIPMENT)}"
            )

    # --- target (training only) ---------------------------------------------------
    if require_target and "posted_rate" in df.columns:
        target = _numeric(df.assign(posted_rate=df["posted_rate"]), "posted_rate", report)
        if target.isna().any():
            report.errors.append(f"posted_rate: {int(target.isna().sum())} missing value(s)")
        if (target <= 0).any():
            report.errors.append(f"posted_rate: {int((target <= 0).sum())} non-positive value(s)")

    # --- warnings (handled by the assessment cleaning layer) ---------------------------
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            report.warnings.append(f"{column}: column absent; treated as missing for all rows")
            continue
        count = int(numeric[column].isna().sum())
        if count:
            rate = count / n
            text = f"{column}_missing_rate={_pct(count, n)}"
            if n >= 20 and rate >= HIGH_MISSING_RATE:
                text += f" (high: >= {int(HIGH_MISSING_RATE * 100)}%)"
            report.warnings.append(text)
    if "weight" in numeric:
        w = numeric["weight"]
        neg = int((w < 0).sum())
        if neg:
            report.warnings.append(f"weight_negative_rate={_pct(neg, n)} (kept raw + flagged by cleaning)")
        cap = int((w.abs() >= WEIGHT_CAP).sum())
        if cap:
            report.warnings.append(f"weight_at_cap_rate={_pct(cap, n)} (|weight| >= {WEIGHT_CAP:,.0f})")

    report.valid = not report.errors
    return report


def validate_record(record: dict[str, Any]) -> ValidationReport:
    return validate_dataframe(pd.DataFrame([record]))


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy ready for the assessment cleaning layer.

    Adds absent optional columns as NaN and coerces numerics. Call only after
    ``validate_dataframe`` reported no errors.
    """
    out = df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    return out
