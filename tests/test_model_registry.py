import json

import pytest

import helpers
from helpers import make_registry, make_metadata, small_artifact, settings_for
from production.model_loader import ModelNotAvailableError, load_production_model
from production.model_registry import (
    REQUIRED_METADATA_KEYS, RegistryError, get_production_model, list_models, promote_model,
    read_production_record, register_model, rollback_model, validate_version,
)


def test_register_creates_version_with_required_metadata(tmp_path):
    reg = tmp_path / "registry"
    artifact, _ = small_artifact()
    meta = register_model(make_metadata(), reg, artifact=artifact)
    assert meta["version"] == "v1"
    assert (reg / "model_v1" / "model.joblib").is_file()
    stored = json.loads((reg / "model_v1" / "metadata.json").read_text())
    for key in REQUIRED_METADATA_KEYS:
        assert key in stored, key
    assert stored["feature_count"] == len(stored["feature_names"]) == 43


def test_register_does_not_promote_and_never_overwrites(tmp_path):
    reg = tmp_path / "registry"
    artifact, _ = small_artifact()
    register_model(make_metadata(), reg, artifact=artifact)
    before = (reg / "model_v1" / "model.joblib").read_bytes()
    second = register_model(make_metadata(alpha=0.5), reg, artifact=artifact)
    assert second["version"] == "v2"
    assert (reg / "model_v1" / "model.joblib").read_bytes() == before
    assert read_production_record(reg) is None  # registration alone never promotes


def test_register_rejects_incomplete_metadata(tmp_path):
    artifact, _ = small_artifact()
    meta = make_metadata()
    del meta["metrics"]
    with pytest.raises(RegistryError):
        register_model(meta, tmp_path / "registry", artifact=artifact)
    assert list_models(tmp_path / "registry") == []


def test_register_requires_exactly_one_source(tmp_path):
    with pytest.raises(RegistryError):
        register_model(make_metadata(), tmp_path / "registry")


def test_promote_writes_production_json(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=False)
    record = promote_model("v2", reg)
    on_disk = json.loads((reg / "production.json").read_text())
    assert on_disk["production_version"] == "v2" and on_disk["status"] == "production"
    assert on_disk["model_name"] == "freight_rate_ridge"
    assert record["previous_version"] is None


def test_promote_rejects_missing_model_file(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=True)
    (reg / "model_v2" / "model.joblib").unlink()
    with pytest.raises(RegistryError, match="model file"):
        promote_model("v2", reg)
    assert read_production_record(reg)["production_version"] == "v1"


def test_promote_rejects_missing_metadata_and_bad_keys(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=True)
    meta_path = reg / "model_v2" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["training_rows"]
    meta_path.write_text(json.dumps(meta))
    assert any("training_rows" in p for p in validate_version("v2", reg))
    meta_path.unlink()
    with pytest.raises(RegistryError):
        promote_model("v2", reg)


def test_promote_rejects_feature_count_mismatch(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=True)
    meta_path = reg / "model_v2" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_count"] = 42
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RegistryError, match="feature"):
        promote_model("v2", reg)


def test_promote_rejects_missing_metrics(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=True)
    meta_path = reg / "model_v2" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["metrics"] = {}
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RegistryError, match="metrics"):
        promote_model("v2", reg)


def test_promote_unknown_version_rejected(tmp_path):
    reg = make_registry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="not registered"):
        promote_model("v9", reg)
    with pytest.raises(RegistryError):
        promote_model("latest", reg)


def test_rollback_to_explicit_version_keeps_all_versions(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=True)
    promote_model("v2", reg)
    record = rollback_model("v1", reg)
    assert record["production_version"] == "v1" and record["previous_version"] == "v2"
    assert (reg / "model_v1").is_dir() and (reg / "model_v2").is_dir()
    assert [h["action"] for h in record["history"]] == ["promote", "promote", "rollback"]


def test_rollback_default_uses_previous_version(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=True)
    promote_model("v2", reg)
    assert rollback_model(None, reg)["production_version"] == "v1"


def test_failed_rollback_leaves_production_untouched(tmp_path):
    reg = make_registry(tmp_path / "registry")
    before = (reg / "production.json").read_text()
    with pytest.raises(RegistryError):
        rollback_model("v7", reg)
    with pytest.raises(RegistryError):
        rollback_model(None, reg)  # no previous version recorded
    assert (reg / "production.json").read_text() == before


def test_list_models_marks_production(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=3, promote=True)
    rows = list_models(reg)
    assert [r["version"] for r in rows] == ["v1", "v2", "v3"]
    assert [r["is_production"] for r in rows] == [True, False, False]
    assert rows[0]["feature_count"] == 43 and rows[0]["MAE"] == 100.0


def test_get_production_model_and_override(tmp_path):
    reg = make_registry(tmp_path / "registry", versions=2, promote=True)
    assert get_production_model(reg)["version"] == "v1"
    assert get_production_model(reg, override_version="v2")["version"] == "v2"
    with pytest.raises(RegistryError):
        get_production_model(reg, override_version="v5")


def test_get_production_model_without_production_json(tmp_path):
    with pytest.raises(RegistryError, match="no production model"):
        get_production_model(tmp_path / "empty")


def test_model_loader_loads_promoted_model_without_training(tmp_path):
    make_registry(tmp_path / "registry")
    loaded = load_production_model(settings_for(tmp_path))
    assert loaded.version == "v1" and len(loaded.feature_columns) == 43


def test_model_loader_respects_version_override_setting(tmp_path):
    make_registry(tmp_path / "registry", versions=2)
    loaded = load_production_model(settings_for(tmp_path, PRODUCTION_MODEL_VERSION="v2"))
    assert loaded.version == "v2"


def test_model_loader_clear_error_when_model_missing(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="no production model"):
        load_production_model(settings_for(tmp_path))


def test_config_reads_environment(tmp_path):
    s = settings_for(tmp_path, LOG_LEVEL="debug", API_PORT="9001", PREDICTION_LOGGING="false")
    assert s.log_level == "DEBUG" and s.api_port == 9001 and s.log_predictions is False
    assert s.registry_path == tmp_path / "registry"
    assert helpers.get_settings({"LOG_LEVEL": "bogus"}).log_level == "INFO"
