"""Evaluate logged production predictions against actual rates, once they exist.

    python scripts/evaluate_production_model.py --actuals actuals.csv     # columns: load_id,actual_rate

Without --actuals (or with no overlap) the report says: "actual target data unavailable".
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse
import json

import pandas as pd

from production.config import get_settings
from production.model_registry import RegistryError, get_production_model
from production.monitoring import read_records, records_to_prediction_frame
from production.performance import UNAVAILABLE_MESSAGE, evaluate_performance, join_predictions_with_actuals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actuals", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()

    try:
        baseline = get_production_model(settings.registry_path, settings.production_model_version)["metadata"].get("metrics")
    except RegistryError:
        baseline = None

    data = None
    if args.actuals:
        predictions = records_to_prediction_frame(read_records(settings.prediction_log_path)[0])
        if not predictions.empty:
            data = join_predictions_with_actuals(predictions, pd.read_csv(args.actuals))
    report = evaluate_performance(data, baseline)
    if not report["available"]:
        report["message"] = UNAVAILABLE_MESSAGE

    out = settings.performance_dir / "performance_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report if report["available"] else {"available": False, "message": report["message"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
