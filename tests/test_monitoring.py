import json
import logging
import math

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from production.logging_config import JsonFormatter, redact
from production.monitoring import (
    NO_DATA_MESSAGE, PredictionMonitor, read_records, records_to_prediction_frame, summarize_predictions,
)
from production.performance import (
    UNAVAILABLE_MESSAGE, compare_to_baseline, compute_metrics, evaluate_performance, join_predictions_with_actuals,
)
from production.retraining import evaluate_retraining_need, read_status, update_status


def log_three(monitor):
    for i, (rate, lat) in enumerate([(1000.0, 10.0), (2000.0, 20.0), (3000.0, 30.0)]):
        monitor.log_prediction(load_id=f"L{i}", predicted_rate=rate, model_version="v1", latency_ms=lat,
                               lower_rate=rate * 0.9, upper_rate=rate * 1.1,
                               inputs={"distance": 500 + i, "equipment": "Reefer", "secret_token": "x", "weight": None})


def test_prediction_record_is_created_with_required_fields(tmp_path):
    path = tmp_path / "p" / "predictions.jsonl"
    monitor = PredictionMonitor(path)
    assert monitor.log_prediction(load_id="L1", predicted_rate=1234.5, model_version="v1", latency_ms=12.4)
    record = json.loads(path.read_text().strip())
    assert {"timestamp", "load_id", "predicted_rate", "model_version", "latency_ms", "event"} <= set(record)
    assert record["predicted_rate"] == 1234.5 and record["model_version"] == "v1" and record["latency_ms"] == 12.4


def test_only_whitelisted_inputs_logged_and_can_be_disabled(tmp_path):
    path = tmp_path / "predictions.jsonl"
    log_three(PredictionMonitor(path))
    record = read_records(path)[0][0]
    assert "secret_token" not in record["inputs"] and record["inputs"]["equipment"] == "Reefer"
    path2 = tmp_path / "noinputs.jsonl"
    log_three(PredictionMonitor(path2, log_inputs=False))
    assert "inputs" not in read_records(path2)[0][0]


def test_disabled_monitor_writes_nothing(tmp_path):
    path = tmp_path / "predictions.jsonl"
    assert PredictionMonitor(path, enabled=False).log_prediction(load_id="a", predicted_rate=1, model_version="v1", latency_ms=1) is False
    assert not path.exists()


def test_unwritable_path_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert PredictionMonitor(blocker / "sub" / "p.jsonl").log_prediction(load_id="a", predicted_rate=1, model_version="v1", latency_ms=1) is False


def test_summary_without_data_says_so(tmp_path):
    summary = summarize_predictions(tmp_path / "missing.jsonl")
    assert summary["available"] is False and summary["message"] == NO_DATA_MESSAGE


def test_summary_statistics(tmp_path):
    path = tmp_path / "predictions.jsonl"
    monitor = PredictionMonitor(path)
    log_three(monitor)
    monitor.log_error(error_type="input_validation", status_code=422, latency_ms=1.0, model_version="v1")
    s = summarize_predictions(path)
    assert s["n_predictions"] == 3 and s["error_count"] == 1
    assert s["mean_prediction"] == 2000.0 and s["median_prediction"] == 2000.0
    assert s["min_prediction"] == 1000.0 and s["max_prediction"] == 3000.0
    assert s["avg_latency_ms"] == 20.0
    assert sum(s["prediction_distribution"]["counts"]) == 3
    assert s["errors_by_type"] == {"input_validation": 1}
    assert s["by_model_version"] == {"v1": 3}


def test_summary_tolerates_malformed_lines(tmp_path):
    path = tmp_path / "predictions.jsonl"
    PredictionMonitor(path).log_prediction(load_id="a", predicted_rate=5.0, model_version="v1", latency_ms=1)
    with open(path, "a") as f:
        f.write("{not json\n\n")
    s = summarize_predictions(path)
    assert s["n_predictions"] == 1 and s["malformed_lines"] == 1


def test_records_to_frame_flattens_inputs(tmp_path):
    path = tmp_path / "predictions.jsonl"
    log_three(PredictionMonitor(path))
    frame = records_to_prediction_frame(read_records(path)[0])
    assert len(frame) == 3 and "distance" in frame.columns and "equipment" in frame.columns


def test_logging_redacts_secrets_and_is_json():
    assert redact({"api_key": "abc", "nested": {"Password": "p", "ok": 1}}) == {
        "api_key": "***REDACTED***", "nested": {"Password": "***REDACTED***", "ok": 1}}
    record = logging.LogRecord("freight_rate.t", logging.INFO, __file__, 1, "hello", (), None)
    record.api_key = "super-secret"
    record.model_version = "v1"
    out = json.loads(JsonFormatter().format(record))
    assert out["message"] == "hello" and out["level"] == "INFO" and out["model_version"] == "v1"
    assert out["api_key"] == "***REDACTED***" and "super-secret" not in json.dumps(out)


# ---------------------------------------------------------------- performance
def test_compute_metrics_known_values():
    m = compute_metrics([100, 200, 400], [110, 180, 400])
    assert m["n"] == 3
    assert math.isclose(m["MAE"], 10.0)
    assert math.isclose(m["RMSE"], math.sqrt((100 + 400 + 0) / 3))
    assert math.isclose(m["MAPE_percent"], (0.10 + 0.10 + 0.0) / 3 * 100)
    assert math.isclose(m["bias_mean_error"], (10 - 20 + 0) / 3)  # predicted - actual


def test_performance_unavailable_without_actuals():
    for data in (None, pd.DataFrame(), pd.DataFrame({"load_id": ["a"], "predicted_rate": [1.0]})):
        report = evaluate_performance(data)
        assert report["available"] is False and report["message"] == UNAVAILABLE_MESSAGE == "actual target data unavailable"


def perf_frame(n=40, scale=1.0):
    rng = np.random.default_rng(1)
    actual = rng.uniform(300, 3000, n)
    return pd.DataFrame({
        "load_id": [f"L{i}" for i in range(n)], "actual_rate": actual, "predicted_rate": actual * scale,
        "equipment": ["Dry Van", "Reefer"] * (n // 2), "distance": rng.uniform(50, 2500, n),
        "date": ["2025-11-05", "2025-12-05"] * (n // 2)})


def test_performance_groups_and_baseline_status():
    base = {"MAE": 100.0, "RMSE": 600.0}
    ok = evaluate_performance(perf_frame(scale=1.0), base)
    assert ok["available"] and ok["overall"]["MAE"] == 0.0 and ok["baseline_comparison"]["status"] == "ok"
    assert {g["group"] for g in ok["by_equipment"]} == {"Dry Van", "Reefer"}
    assert {g["group"] for g in ok["by_month"]} == {"2025-11", "2025-12"}
    assert ok["by_distance_bucket"]
    bad = evaluate_performance(perf_frame(scale=1.5), {"MAE": 100.0})
    assert bad["baseline_comparison"]["status"] == "degraded"


def test_baseline_comparison_statuses():
    cur = lambda mae, n=100: {"MAE": mae, "RMSE": 1.0, "n": n}
    assert compare_to_baseline(cur(105), {"MAE": 100})["status"] == "ok"
    assert compare_to_baseline(cur(115), {"MAE": 100})["status"] == "warning"
    assert compare_to_baseline(cur(130), {"MAE": 100})["status"] == "degraded"
    assert compare_to_baseline(cur(130, n=5), {"MAE": 100})["status"] == "insufficient_data"
    assert compare_to_baseline(cur(130), None)["status"] == "no_baseline"


def test_join_predictions_with_actuals_last_wins():
    preds = pd.DataFrame({"load_id": ["a", "a", "b"], "predicted_rate": [1.0, 2.0, 3.0]})
    actuals = pd.DataFrame({"load_id": ["a", "c"], "actual_rate": [5.0, 6.0]})
    joined = join_predictions_with_actuals(preds, actuals)
    assert joined["load_id"].tolist() == ["a"] and joined["predicted_rate"].tolist() == [2.0]


# ---------------------------------------------------------------- retraining policy
def test_retraining_policy_reasons(tmp_path):
    drift = {"features": [{"feature": "distance", "status": "drift"}, {"feature": "weight", "status": "drift"}]}
    perf = {"available": True, "baseline_comparison": {"status": "degraded", "mae_change_percent": 40.0}}
    meta = {"training_date": "2020-01-01T00:00:00+00:00"}
    d = evaluate_retraining_need(drift, perf, meta)
    assert d["recommended"] and len(d["reasons"]) == 3
    quiet = evaluate_retraining_need(None, None, None)
    assert quiet["recommended"] is False and quiet["note"]
    one = evaluate_retraining_need({"features": [{"feature": "distance", "status": "drift"}]}, None, None)
    assert one["recommended"] is False  # a single drifted feature is not enough


def test_retraining_status_roundtrip(tmp_path):
    assert read_status(tmp_path) == {}
    update_status(tmp_path, last_training={"version": "v2"})
    update_status(tmp_path, last_check={"recommended": False})
    status = read_status(tmp_path)
    assert status["last_training"]["version"] == "v2" and status["last_check"]["recommended"] is False
