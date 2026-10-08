"""Production training / tuning, built on the existing assessment functions.

Reused as-is from the assessment code: ``ridge_tuning.temporal_split``,
``ridge_tuning.prepare_features`` (Jan-Aug train / Sep-Oct validation, cleaning fitted
on the training fold only), ``data_cleaning`` and ``features``. The model is the same
Pipeline(median imputer -> StandardScaler -> Ridge) trained on log1p(rate).
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import production  # noqa: F401  (adds src/ to sys.path)
from data_cleaning import apply_cleaning, fit_cleaning_statistics
from features import build_features, get_model_features
from ridge_tuning import evaluate as _evaluate_clipped
from ridge_tuning import prepare_features, temporal_split

from production.config import Settings, get_settings
from production.drift import build_reference_profile
from production.logging_config import get_logger
from production.model_registry import get_git_commit, register_model
from production.retraining import update_status
from production.uncertainty import compute_interval_metadata
from production.validation import InputValidationError, validate_dataframe

log = get_logger("training")

SEED = 42  # the pipeline is deterministic (no stochastic components); recorded for reproducibility
ALPHA_GRID = (0.001, 0.01, 0.1, 1, 10, 100)
TARGET_TRANSFORM = "log1p"
INVERSE_TRANSFORM = "expm1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(alpha: float) -> Pipeline:
    """Identical structure to src/final_pipeline.py."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def load_training_data(path: Path) -> pd.DataFrame:
    """Read and *validate* the labelled development data (raises on fundamental problems)."""
    df = pd.read_csv(path)
    report = validate_dataframe(df, require_target=True)
    if not report.valid:
        raise InputValidationError(report)
    log.info("training data validated", extra={"event": "training_data_validated", "rows": report.rows, "warnings": report.warnings})
    return df


def _metrics(y_true, y_pred) -> dict:
    mae, rmse, r2 = _evaluate_clipped(y_true, y_pred)  # same clipping + formulas as the assessment
    y = np.asarray(y_true, dtype=float)
    p = np.maximum(np.asarray(y_pred, dtype=float), 1.0)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
            "MAPE_percent": float(np.mean(np.abs(y - p) / y) * 100.0)}


def evaluate_alphas(df: pd.DataFrame, alphas: Sequence[float] = ALPHA_GRID) -> tuple[pd.DataFrame, dict]:
    """Temporal-validation sweep. Returns (results sorted by MAE, holdout detail per alpha)."""
    np.random.seed(SEED)
    train, valid = temporal_split(df)
    X_train, X_valid, y_train, y_valid, _ = prepare_features(train, valid)
    rows, detail = [], {}
    for alpha in alphas:
        model = build_pipeline(float(alpha)).fit(X_train, np.log1p(y_train))
        pred = np.expm1(model.predict(X_valid))
        rows.append({"alpha": float(alpha), **_metrics(y_valid, pred)})
        detail[float(alpha)] = {"y_true": y_valid.to_numpy(), "y_pred": pred, "dates": valid["date"].to_numpy()}
    results = pd.DataFrame(rows).sort_values(["MAE", "alpha"]).reset_index(drop=True)
    return results, detail


def select_best_alpha(results: pd.DataFrame) -> float:
    """Lowest MAE wins; exact ties go to the smaller alpha."""
    return float(results.sort_values(["MAE", "alpha"]).iloc[0]["alpha"])


def fit_final_artifact(df: pd.DataFrame, alpha: float) -> dict:
    """Train on ALL labelled rows. Same artifact structure as src/final_pipeline.py."""
    stats = fit_cleaning_statistics(df)
    features = build_features(apply_cleaning(df, stats))
    columns = get_model_features(features)
    model = build_pipeline(alpha).fit(features[columns], np.log1p(features["posted_rate"]))
    return {"model": model, "feature_columns": columns, "cleaning_stats": stats}


def build_metadata(*, df: pd.DataFrame, artifact: dict, alpha: float, holdout_metrics: dict,
                   interval: dict, reference_profile: dict, data_path: Path, tuning: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["date"])
    return {
        "algorithm": "Ridge",
        "model": "Ridge",
        "alpha": float(alpha),
        "training_rows": int(len(df)),
        "feature_count": len(artifact["feature_columns"]),
        "feature_names": list(artifact["feature_columns"]),
        "training_date": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "training_data_period": {"start": str(dates.min().date()), "end": str(dates.max().date())},
        "target_transformation": TARGET_TRANSFORM,
        "inverse_transformation": INVERSE_TRANSFORM,
        "metrics": {**holdout_metrics,
                    "evaluation": "temporal holdout: trained Jan-Aug 2025, validated Sep-Oct 2025 (same split as the assessment)"},
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "git_commit": get_git_commit(),
        "random_seed": SEED,
        "training_data_file": data_path.name,
        "training_data_sha256": sha256_file(data_path),
        "tuning_grid": [float(a) for a in tuning["alpha"]],
        "prediction_interval": interval,
        "reference_profile": reference_profile,
    }


def run_training_pipeline(settings: Optional[Settings] = None, *, data_path: Optional[Path] = None,
                          alphas: Sequence[float] = ALPHA_GRID, coverage: float = 0.90,
                          registry_path: Optional[Path] = None) -> dict:
    """load -> validate -> clean -> features -> temporal eval -> select alpha ->
    train final -> metrics -> register (NOT promoted)."""
    settings = settings or get_settings()
    data_path = Path(data_path or settings.training_data_path)
    registry = Path(registry_path or settings.registry_path)

    df = load_training_data(data_path)
    results, detail = evaluate_alphas(df, alphas)
    alpha = select_best_alpha(results)
    best = results[results["alpha"] == alpha].iloc[0]
    holdout = {k: float(best[k]) for k in ("MAE", "RMSE", "R2", "MAPE_percent")}
    holdout["validation_rows"] = int(len(detail[alpha]["y_true"]))

    interval = compute_interval_metadata(
        detail[alpha]["y_true"], detail[alpha]["y_pred"], dates=detail[alpha]["dates"], coverage=coverage,
        source="residuals of the temporal-holdout model (trained Jan-Aug 2025, scored Sep-Oct 2025)",
    )
    profile = build_reference_profile(df, source=f"{data_path.name} ({len(df):,} rows)")
    artifact = fit_final_artifact(df, alpha)
    metadata = build_metadata(df=df, artifact=artifact, alpha=alpha, holdout_metrics=holdout, interval=interval,
                              reference_profile=profile, data_path=data_path, tuning=results)
    registered = register_model(metadata, registry, artifact=artifact)

    update_status(settings.retraining_dir, last_training={
        "version": registered["version"], "trained_at": registered["training_date"], "alpha": alpha,
        "MAE": holdout["MAE"], "promoted": False,
        "note": "registered only; promotion is a separate explicit step (scripts/promote_model.py)",
    })
    log.info("training complete", extra={"event": "training_complete", "model_version": registered["version"], "alpha": alpha})
    return {"metadata": registered, "tuning": results}
