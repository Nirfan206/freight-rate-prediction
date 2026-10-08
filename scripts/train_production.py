"""Retrain, evaluate and REGISTER a new model version (never auto-promotes).

    python scripts/train_production.py                 # tune over the default grid, register next version
    python scripts/train_production.py --alphas 0.01   # pin alpha (reproduces the assessment model)
    python scripts/train_production.py --check-only    # only report whether retraining is recommended
    python scripts/train_production.py --if-needed     # train only if the retraining policy recommends it

Promotion is a separate, explicit step: python scripts/promote_model.py --version vN
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse
import json

from production.config import get_settings
from production.logging_config import configure_logging
from production.model_registry import RegistryError, get_production_model
from production.retraining import evaluate_retraining_need, update_status
from production.training import ALPHA_GRID, run_training_pipeline


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except json.JSONDecodeError:
        return None


def retraining_check(settings) -> dict:
    try:
        metadata = get_production_model(settings.registry_path, settings.production_model_version)["metadata"]
    except RegistryError:
        metadata = None
    decision = evaluate_retraining_need(
        _load_json(settings.drift_dir / "drift_report.json"),
        _load_json(settings.performance_dir / "performance_report.json"),
        metadata,
    )
    update_status(settings.retraining_dir, last_check=decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="labelled CSV (default: data/train-test.csv)")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHA_GRID))
    parser.add_argument("--coverage", type=float, default=0.90, help="nominal prediction-interval coverage")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--if-needed", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.check_only or args.if_needed:
        decision = retraining_check(settings)
        print(json.dumps(decision, indent=2))
        if args.check_only or not decision["recommended"]:
            return 0

    outcome = run_training_pipeline(settings, data_path=args.data, alphas=args.alphas, coverage=args.coverage)
    meta = outcome["metadata"]
    print(outcome["tuning"].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    m = meta["metrics"]
    print(f"\nRegistered {meta['version']}: alpha={meta['alpha']} rows={meta['training_rows']:,} features={meta['feature_count']}")
    print(f"Temporal holdout: MAE={m['MAE']:.6f} RMSE={m['RMSE']:.6f} R2={m['R2']:.6f} MAPE={m['MAPE_percent']:.6f}%")
    iv = meta["prediction_interval"]
    chk = iv.get("holdout_check") or {}
    print(f"Interval ({iv['nominal_coverage']:.0%} nominal): out-of-sample coverage on later holdout = {chk.get('empirical_coverage')}")
    print(f"NOT promoted. To promote: python scripts/promote_model.py --version {meta['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
