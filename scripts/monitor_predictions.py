"""Summarise the production prediction log.

    python scripts/monitor_predictions.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse
import json

from production.config import get_settings
from production.monitoring import summarize_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()
    summary = summarize_predictions(args.log or settings.prediction_log_path)
    out = settings.monitoring_path / "predictions" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
