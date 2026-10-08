"""Deterministic Ridge alpha search using the assessment's temporal validation.

    python scripts/tune_model.py

Writes outputs/experiments/production_tuning.csv (alpha, MAE, RMSE, R2, MAPE_percent).
Does not touch the assessment's ridge_tuning_results.csv or any model.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse

from production.config import get_settings
from production.logging_config import configure_logging
from production.training import ALPHA_GRID, evaluate_alphas, load_training_data, select_best_alpha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHA_GRID))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "experiments" / "production_tuning.csv")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    df = load_training_data(settings.training_data_path)
    results, _ = evaluate_alphas(df, args.alphas)
    best = select_best_alpha(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nBest alpha by MAE: {best}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
