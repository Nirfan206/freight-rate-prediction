"""Compare input data with the production model's training reference.

    python scripts/detect_drift.py                          # inputs from the prediction log
    python scripts/detect_drift.py --input data/validation.csv   # any CSV of raw inputs

The report states its source. A CSV comparison is NOT production traffic.
Thresholds: PSI < 0.10 stable, 0.10-0.25 warning, > 0.25 drift.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse
import json

import pandas as pd

from production.config import get_settings
from production.drift import detect_drift
from production.model_registry import RegistryError, get_production_model
from production.monitoring import NO_DATA_MESSAGE, read_records, records_to_prediction_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="CSV with raw input columns")
    args = parser.parse_args()
    settings = get_settings()

    try:
        profile = get_production_model(settings.registry_path, settings.production_model_version)["metadata"].get("reference_profile")
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not profile:
        print("ERROR: production model metadata has no reference_profile", file=sys.stderr)
        return 1

    if args.input:
        current, source = pd.read_csv(args.input), f"csv:{args.input.name}"
    else:
        records, _ = read_records(settings.prediction_log_path)
        current, source = records_to_prediction_frame(records), "production prediction log"
        if current.empty:
            print(NO_DATA_MESSAGE)
            return 0

    report = detect_drift(current, profile, source=source)
    out = settings.drift_dir / "drift_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"source={source} rows={report['n_current_rows']:,} overall={report['overall_status']}")
    for f in report["features"]:
        value = "n/a" if f["value"] is None else f"{f['value']:.4f}"
        print(f"  {f['feature']:<14} {f['metric']:<16} {value:>10}  {f['status']}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
