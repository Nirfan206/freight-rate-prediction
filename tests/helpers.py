"""Shared helpers for production tests (plain functions - no fixtures needed)."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from production.config import get_settings  # noqa: E402
from production.drift import build_reference_profile  # noqa: E402
from production.model_registry import promote_model, register_model  # noqa: E402
from production.training import fit_final_artifact  # noqa: E402
from production.uncertainty import compute_interval_metadata  # noqa: E402
from production.validation import SMOKE_RECORD  # noqa: E402

TRAIN_CSV = ROOT / "data" / "train-test.csv"
VALID_RECORD = dict(SMOKE_RECORD, load_id="LOAD_001")


def training_frame() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


@lru_cache(maxsize=1)
def small_artifact():
    """A quickly trained artifact (first 6,000 rows, alpha=0.01) shared across tests."""
    df = training_frame().iloc[:6000].copy()
    return fit_final_artifact(df, 0.01), df


def make_metadata(**overrides) -> dict:
    artifact, df = small_artifact()
    rng = np.random.default_rng(0)
    actual = rng.uniform(300, 3000, 400)
    pred = actual * np.exp(rng.normal(0, 0.05, 400))
    meta = {
        "algorithm": "Ridge", "model": "Ridge", "alpha": 0.01, "training_rows": len(df),
        "feature_count": len(artifact["feature_columns"]), "feature_names": list(artifact["feature_columns"]),
        "training_data_period": {"start": "2025-01-01", "end": "2025-02-28"},
        "target_transformation": "log1p", "inverse_transformation": "expm1",
        "metrics": {"MAE": 100.0, "RMSE": 600.0, "R2": 0.8, "MAPE_percent": 4.8},
        "prediction_interval": compute_interval_metadata(actual, pred, coverage=0.9),
        "reference_profile": build_reference_profile(df),
    }
    meta.update(overrides)
    return meta


def make_registry(path: Path, versions: int = 1, promote: bool = True) -> Path:
    """Create a registry with N identical-structure versions; promote v1 if asked."""
    artifact, _ = small_artifact()
    for _ in range(versions):
        register_model(make_metadata(), path, artifact=artifact)
    if promote:
        promote_model("v1", path)
    return path


def settings_for(tmp_path: Path, **env):
    base = {"MODEL_REGISTRY_PATH": str(tmp_path / "registry"), "MONITORING_PATH": str(tmp_path / "monitoring")}
    base.update(env)
    return get_settings(base)
