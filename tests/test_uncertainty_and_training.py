import numpy as np
import pandas as pd

import helpers  # noqa: F401
from helpers import training_frame
from production.config import get_settings
from production.model_registry import list_models, read_production_record, REQUIRED_METADATA_KEYS
from production.retraining import read_status
from production.training import (
    ALPHA_GRID, evaluate_alphas, run_training_pipeline, select_best_alpha,
)
from production.uncertainty import DISCLAIMER, apply_interval, compute_interval_metadata


def noisy(n=20000, sigma=0.08, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.uniform(300, 4000, n)
    pred = actual * np.exp(rng.normal(0, sigma, n))
    dates = pd.date_range("2025-09-01", periods=n, freq="min")
    return actual, pred, dates


def test_interval_ordering_and_nonnegative():
    actual, pred, _ = noisy()
    meta = compute_interval_metadata(actual, pred, coverage=0.9)
    lo, hi = apply_interval(np.array([100.0, 1000.0, 5000.0]), meta)
    p = np.array([100.0, 1000.0, 5000.0])
    assert (lo <= p).all() and (p <= hi).all() and (lo >= 0).all()


def test_interval_is_multiplicative_wider_for_expensive_loads():
    actual, pred, _ = noisy()
    meta = compute_interval_metadata(actual, pred, coverage=0.9)
    lo, hi = apply_interval(np.array([500.0, 5000.0]), meta)
    assert (hi[1] - lo[1]) > 5 * (hi[0] - lo[0])


def test_interval_coverage_close_to_nominal_on_fresh_data():
    actual, pred, dates = noisy(seed=0)
    meta = compute_interval_metadata(actual, pred, dates=dates, coverage=0.9)
    a2, p2, _ = noisy(seed=99)
    lo, hi = apply_interval(p2, meta)
    coverage = float(np.mean((a2 >= lo) & (a2 <= hi)))
    assert abs(coverage - 0.9) < 0.02
    check = meta["holdout_check"]
    assert abs(check["empirical_coverage"] - 0.9) < 0.03 and check["n_evaluation"] > 50


def test_interval_metadata_states_it_is_not_calibrated():
    actual, pred, _ = noisy(n=500)
    meta = compute_interval_metadata(actual, pred)
    assert meta["disclaimer"] == DISCLAIMER
    assert "not calibrated" in meta["disclaimer"] and meta["method"] == "empirical_log_residual_quantiles"


def test_no_interval_metadata_gives_degenerate_interval():
    lo, hi = apply_interval([1000.0], None)
    assert lo[0] == hi[0] == 1000.0


def test_alpha_grid_matches_spec():
    assert ALPHA_GRID == (0.001, 0.01, 0.1, 1, 10, 100)


def test_alpha_sweep_reproduces_assessment_alpha_001_metrics():
    results, _ = evaluate_alphas(training_frame(), [0.01])
    row = results.iloc[0]
    assert abs(row["MAE"] - 108.607337) < 1e-5
    assert abs(row["RMSE"] - 633.255669) < 1e-5
    assert abs(row["R2"] - 0.827805) < 1e-5
    assert abs(row["MAPE_percent"] - 4.811586) < 1e-5


def test_select_best_alpha_by_mae_and_tie_break():
    results = pd.DataFrame({"alpha": [1.0, 0.01, 0.1], "MAE": [5.0, 3.0, 3.0], "RMSE": [1, 1, 1], "R2": [0, 0, 0]})
    assert select_best_alpha(results) == 0.01


def test_training_pipeline_registers_but_does_not_promote(tmp_path):
    settings = get_settings({"MODEL_REGISTRY_PATH": str(tmp_path / "registry"), "MONITORING_PATH": str(tmp_path / "mon")})
    outcome = run_training_pipeline(settings, alphas=[0.01, 1.0])
    meta = outcome["metadata"]
    assert meta["version"] == "v1" and meta["alpha"] == 0.01 and meta["training_rows"] == 48000
    for key in REQUIRED_METADATA_KEYS:
        assert key in meta
    assert meta["feature_count"] == 43 and meta["prediction_interval"]["disclaimer"]
    assert read_production_record(settings.registry_path) is None
    assert [m["version"] for m in list_models(settings.registry_path)] == ["v1"]
    status = read_status(settings.retraining_dir)
    assert status["last_training"]["version"] == "v1" and status["last_training"]["promoted"] is False
    again = run_training_pipeline(settings, alphas=[0.01])
    assert again["metadata"]["version"] == "v2"  # a second run never overwrites v1
