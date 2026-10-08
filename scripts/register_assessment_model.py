"""Register the EXISTING assessment artifact (artifacts/ridge_model.joblib) as a registry version.

    python scripts/register_assessment_model.py            # registers (does not promote)
    python scripts/register_assessment_model.py --promote  # ...and promotes it (with smoke test)

The assessment model file is copied byte-for-byte and never modified. Metrics, prediction
interval and drift reference are recomputed from the same temporal split (a fresh fit on
Jan-Aug, scored on Sep-Oct) - nothing is unpickled to obtain them.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse
import json
import re
from datetime import datetime, timezone

import pandas as pd

from production.config import get_settings
from production.drift import build_reference_profile
from production.logging_config import configure_logging
from production.model_registry import RegistryError, list_models, promote_model, register_model
from production.training import (
    INVERSE_TRANSFORM, SEED, TARGET_TRANSFORM, evaluate_alphas, load_training_data, sha256_file,
)
from production.uncertainty import compute_interval_metadata


def pinned_sklearn_version() -> str | None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"scikit-learn==([\w.]+)", text)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--coverage", type=float, default=0.90)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    model_file = ROOT / "artifacts" / "ridge_model.joblib"
    meta_file = ROOT / "artifacts" / "model_metadata.json"
    assessment = json.loads(meta_file.read_text(encoding="utf-8"))

    existing = [m for m in list_models(settings.registry_path) if m.get("alpha") == assessment["alpha"]]
    sha = sha256_file(model_file)
    if existing:
        print(f"A version with alpha={assessment['alpha']} is already registered; not registering a duplicate.")
        return 0

    df = load_training_data(settings.training_data_path)
    results, detail = evaluate_alphas(df, [assessment["alpha"]])
    row = results.iloc[0]
    d = detail[float(assessment["alpha"])]
    holdout = {k: float(row[k]) for k in ("MAE", "RMSE", "R2", "MAPE_percent")}
    holdout["validation_rows"] = int(len(d["y_true"]))
    dates = pd.to_datetime(df["date"])

    metadata = {
        "algorithm": assessment["model"], "model": assessment["model"], "alpha": assessment["alpha"],
        "training_rows": assessment["training_rows"], "feature_count": assessment["feature_count"],
        "feature_names": assessment["features"],
        "training_date": datetime.fromtimestamp(model_file.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
        "training_date_source": "artifact file modification time (original training time was not recorded)",
        "training_data_period": {"start": str(dates.min().date()), "end": str(dates.max().date())},
        "target_transformation": assessment["target_transform"],
        "inverse_transformation": assessment["inverse_transform"],
        "metrics": {**holdout, "evaluation": "temporal holdout: trained Jan-Aug 2025, validated Sep-Oct 2025 (same split as the assessment)"},
        "sklearn_version": pinned_sklearn_version(),
        "sklearn_version_source": "requirements.txt pin (the pickle does not record it)",
        "git_commit": None,
        "random_seed": SEED,
        "source": "assessment artifact artifacts/ridge_model.joblib, copied unchanged",
        "model_file_sha256": sha,
        "training_data_file": settings.training_data_path.name,
        "training_data_sha256": sha256_file(settings.training_data_path),
        "prediction_interval": compute_interval_metadata(
            d["y_true"], d["y_pred"], dates=d["dates"], coverage=args.coverage,
            source="residuals of the temporal-holdout model (trained Jan-Aug 2025, scored Sep-Oct 2025)"),
        "reference_profile": build_reference_profile(df, source=f"{settings.training_data_path.name} ({len(df):,} rows)"),
    }
    from production.model_registry import get_git_commit
    metadata["git_commit"] = get_git_commit()
    registered = register_model(metadata, settings.registry_path, model_file=model_file)
    print(f"Registered {registered['version']} from the assessment artifact (sha256 {sha[:12]}...)")
    print(f"  holdout MAE={holdout['MAE']:.6f} RMSE={holdout['RMSE']:.6f} R2={holdout['R2']:.6f} MAPE={holdout['MAPE_percent']:.6f}%")

    if args.promote:
        try:
            promote_model(registered["version"], settings.registry_path, reason="initial production model (assessment artifact)")
        except RegistryError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Promoted {registered['version']} to production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
