"""API tests.

Two layers:
  * service-level tests exercise the exact request logic the endpoints call (no FastAPI needed);
  * HTTP-level tests use FastAPI's TestClient and are skipped only when fastapi/httpx are not installed.
"""

import json

import pytest

import helpers
from helpers import VALID_RECORD, make_registry, settings_for
from production.model_loader import ModelNotAvailableError
from production.monitoring import read_records
from production.validation import InputValidationError

import api  # noqa: F401
from api.service import ModelService


def ready_service(tmp_path, **env):
    make_registry(tmp_path / "registry")
    settings = settings_for(tmp_path, **env)
    service = ModelService(settings)
    assert service.ensure_loaded()
    return service, settings


# ------------------------------------------------------------------ service level
def test_service_predicts_from_raw_input_and_logs(tmp_path):
    service, settings = ready_service(tmp_path)
    result = service.predict_one(dict(VALID_RECORD))
    assert result["load_id"] == "LOAD_001" and result["model_version"] == "v1"
    assert result["predicted_rate"] > 0
    assert result["lower_rate"] <= result["predicted_rate"] <= result["upper_rate"]
    assert result["interval_coverage"] == 0.9
    records, _ = read_records(settings.prediction_log_path)
    assert len(records) == 1 and records[0]["event"] == "prediction" and records[0]["latency_ms"] > 0


def test_service_handles_missing_optional_values(tmp_path):
    service, _ = ready_service(tmp_path)
    record = {k: v for k, v in VALID_RECORD.items() if k not in ("weight", "market_index", "quote_signal")}
    result = service.predict_one(record)
    assert result["predicted_rate"] > 0 and len(result["notes"]) == 3


def test_service_negative_weight_is_handled_not_rejected(tmp_path):
    service, _ = ready_service(tmp_path)
    result = service.predict_one(dict(VALID_RECORD, weight=-4000.0))
    assert result["predicted_rate"] > 0 and any("negative" in n for n in result["notes"])


def test_service_rejects_invalid_input_and_counts_error(tmp_path):
    service, settings = ready_service(tmp_path)
    with pytest.raises(InputValidationError):
        service.predict_one(dict(VALID_RECORD, distance=-1))
    records, _ = read_records(settings.prediction_log_path)
    assert [r["event"] for r in records] == ["error"] and records[0]["error_type"] == "input_validation"
    assert "distance" not in json.dumps(records[0])  # no request payload in error records


def test_service_without_model_raises_clear_error(tmp_path):
    service = ModelService(settings_for(tmp_path))
    assert service.ensure_loaded() is False
    with pytest.raises(ModelNotAvailableError):
        service.predict_one(dict(VALID_RECORD))
    assert "no production model" in service.last_error


def test_service_recovers_when_model_appears_later(tmp_path):
    service = ModelService(settings_for(tmp_path))
    assert service.ensure_loaded() is False
    make_registry(tmp_path / "registry")
    assert service.ensure_loaded() is True


def test_service_model_info(tmp_path):
    service, _ = ready_service(tmp_path)
    info = service.model_info()
    assert info["model"] == "Ridge" and info["version"] == "v1" and info["alpha"] == 0.01
    assert info["feature_count"] == 43 and info["target_transform"] == "log1p"
    assert "not calibrated" in info["prediction_interval"]["disclaimer"]


def test_prediction_logging_can_be_turned_off(tmp_path):
    service, settings = ready_service(tmp_path, PREDICTION_LOGGING="false")
    service.predict_one(dict(VALID_RECORD))
    assert not settings.prediction_log_path.exists()


# ------------------------------------------------------------------ HTTP level
def make_client(tmp_path, with_model=True, **client_kwargs):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.main import create_app

    if with_model:
        make_registry(tmp_path / "registry")
    settings = settings_for(tmp_path)
    app = create_app(settings=settings, service=ModelService(settings))
    return TestClient(app, **client_kwargs), app


def test_http_health_ok(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_http_health_reports_unhealthy_without_model(tmp_path):
    client, _ = make_client(tmp_path, with_model=False)
    with client:
        response = client.get("/health")
        assert response.status_code == 503 and response.json() == {"status": "unhealthy", "model_loaded": False}
        assert client.get("/model/info").status_code == 503
        predict = client.post("/predict", json=VALID_RECORD)
        assert predict.status_code == 503 and "not available" in predict.json()["message"]
        assert "Traceback" not in predict.text


def test_http_model_info(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        body = client.get("/model/info").json()
    assert body["model"] == "Ridge" and body["version"] == "v1" and body["alpha"] == 0.01
    assert body["feature_count"] == 43 and body["target_transform"] == "log1p"


def test_http_valid_prediction(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 200
    body = response.json()
    assert body["load_id"] == "LOAD_001" and body["model_version"] == "v1"
    assert body["lower_rate"] <= body["predicted_rate"] <= body["upper_rate"]


def test_http_missing_optional_values_ok(tmp_path):
    client, _ = make_client(tmp_path)
    payload = {k: v for k, v in VALID_RECORD.items() if k not in ("weight", "market_index", "quote_signal")}
    with client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200 and len(response.json()["notes"]) == 3


def test_http_rejects_bad_inputs(tmp_path):
    client, _ = make_client(tmp_path)
    cases = {
        "distance": dict(VALID_RECORD, distance=0),
        "pickup_lat": dict(VALID_RECORD, pickup_lat=95),
        "delivery_lon": dict(VALID_RECORD, delivery_lon=-181),
        "equipment": dict(VALID_RECORD, equipment="Tanker"),
        "date": dict(VALID_RECORD, date="2025-13-45"),
        "load_id": {k: v for k, v in VALID_RECORD.items() if k != "load_id"},
    }
    with client:
        for field, payload in cases.items():
            response = client.post("/predict", json=payload)
            assert response.status_code == 422, field
            body = response.json()
            assert body["error"] == "validation_error"
            assert any(field in d["field"] for d in body["details"]), (field, body)
            assert "Traceback" not in response.text


def test_http_rejects_unknown_fields_and_nan(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        assert client.post("/predict", json=dict(VALID_RECORD, wieght=1)).status_code == 422
        raw = json.dumps(VALID_RECORD).replace('"distance": 850.0', '"distance": NaN')
        assert client.post("/predict", content=raw, headers={"content-type": "application/json"}).status_code == 422


def test_http_internal_error_hides_details(tmp_path):
    client, app = make_client(tmp_path, raise_server_exceptions=False)
    with client:
        def boom(_records):
            raise RuntimeError("secret internal detail /srv/path")
        app.state.service.predictor.predict_records = boom
        response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"
    assert "secret internal detail" not in response.text and "Traceback" not in response.text


def test_http_docs_and_openapi_available(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        spec = client.get("/openapi.json").json()
    assert {"/health", "/model/info", "/predict"} <= set(spec["paths"])
