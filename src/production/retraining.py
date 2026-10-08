"""Retraining policy and status tracking.

Retraining is *recommended* from evidence (drift, degraded performance, model age) and
is always executed explicitly (scripts/train_production.py). A newly trained model is
registered but never auto-promoted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MAX_MODEL_AGE_DAYS = 90
MIN_DRIFTED_FEATURES = 2

NO_HISTORY_MESSAGE = "No retraining history recorded."


def _now() -> datetime:
    return datetime.now(timezone.utc)


def evaluate_retraining_need(
    drift_report: Optional[dict],
    performance_report: Optional[dict],
    model_metadata: Optional[dict],
    *,
    now: Optional[datetime] = None,
) -> dict:
    now = now or _now()
    reasons: list[str] = []
    evidence = {"drift": bool(drift_report), "performance": bool(performance_report and performance_report.get("available")),
                "model_age": False}

    if drift_report:
        drifted = [f["feature"] for f in drift_report.get("features", []) if f.get("status") == "drift"]
        if len(drifted) >= MIN_DRIFTED_FEATURES:
            reasons.append(f"data drift on {len(drifted)} features: {', '.join(drifted)}")
    if evidence["performance"]:
        comparison = performance_report.get("baseline_comparison") or {}
        if comparison.get("status") == "degraded":
            reasons.append(f"MAE {comparison['mae_change_percent']:+.1f}% vs baseline (degraded)")
    trained = (model_metadata or {}).get("training_date")
    if trained:
        try:
            ts = datetime.fromisoformat(str(trained).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            evidence["model_age"] = True
            age = (now - ts).days
            if age > MAX_MODEL_AGE_DAYS:
                reasons.append(f"model is {age} days old (limit {MAX_MODEL_AGE_DAYS})")
        except ValueError:
            pass

    return {
        "recommended": bool(reasons),
        "reasons": reasons,
        "evidence_available": evidence,
        "evaluated_at": now.isoformat(timespec="seconds"),
        "note": "" if any(evidence.values()) else "no monitoring evidence available yet",
    }


def status_path(retraining_dir: Path) -> Path:
    return Path(retraining_dir) / "status.json"


def read_status(retraining_dir: Path) -> dict:
    path = status_path(retraining_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def update_status(retraining_dir: Path, **sections: dict) -> dict:
    """Merge sections such as last_training / last_check into status.json."""
    status = read_status(retraining_dir)
    status.update(sections)
    path = status_path(retraining_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
