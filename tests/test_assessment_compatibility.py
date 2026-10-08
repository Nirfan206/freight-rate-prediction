"""Guards that the production layer leaves the assessment deliverables intact and consistent."""

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import helpers
from helpers import ROOT, training_frame
from production.model_loader import load_production_model
from production.predictor import Predictor
from production.config import get_settings
from production.model_loader import LoadedModel
from production.training import fit_final_artifact


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_production_pipeline_reproduces_assessment_validation_predictions():
    """Train with the production code path, predict data/validation.csv RAW through the Predictor
    (validation -> cleaning -> features -> model -> expm1) and compare with the official CSV."""
    artifact = fit_final_artifact(training_frame(), 0.01)
    loaded = LoadedModel("test", artifact["model"], artifact["feature_columns"], artifact["cleaning_stats"], {})
    validation = pd.read_csv(ROOT / "data" / "validation.csv")
    produced = Predictor(loaded).predict_frame(validation)
    official = pd.read_csv(ROOT / "validation_predictions.csv")
    assert produced["load_id"].tolist() == official["load_id"].tolist()
    assert np.max(np.abs(produced["predicted_rate"].to_numpy() - official["predicted_rate"].to_numpy())) < 0.01


def test_registry_v1_is_the_unmodified_assessment_artifact():
    v1 = ROOT / "models" / "registry" / "model_v1"
    assert sha(v1 / "model.joblib") == sha(ROOT / "artifacts" / "ridge_model.joblib")
    meta = json.loads((v1 / "metadata.json").read_text())
    assessment = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    assert meta["alpha"] == assessment["alpha"] == 0.01
    assert meta["feature_names"] == assessment["features"] and meta["feature_count"] == 43
    assert meta["training_rows"] == 48000 and meta["target_transformation"] == "log1p"
    assert abs(meta["metrics"]["MAE"] - 108.607337) < 1e-5
    record = json.loads((ROOT / "models" / "registry" / "production.json").read_text())
    assert record["status"] == "production" and record["model_name"] == "freight_rate_ridge"


def test_registry_v1_predictions_match_official_file_when_runtime_matches_training_sklearn():
    import sklearn

    settings = get_settings({})
    located_meta = json.loads((ROOT / "models" / "registry" / "model_v1" / "metadata.json").read_text())
    if located_meta["sklearn_version"] != sklearn.__version__:
        pytest.skip(f"v1 was trained with scikit-learn {located_meta['sklearn_version']}, runtime has {sklearn.__version__}")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no unpickle version warnings expected in the pinned environment
        loaded = load_production_model(get_settings({"PRODUCTION_MODEL_VERSION": "v1"}))
    validation = pd.read_csv(ROOT / "data" / "validation.csv").head(500)
    produced = Predictor(loaded).predict_frame(validation)
    official = pd.read_csv(ROOT / "validation_predictions.csv").head(500)
    assert np.max(np.abs(produced["predicted_rate"].to_numpy() - official["predicted_rate"].to_numpy())) < 1e-6


def test_official_prediction_format_unchanged():
    df = pd.read_csv(ROOT / "validation_predictions.csv")
    assert df.columns.tolist() == ["load_id", "predicted_rate"] and len(df) == 12000


def test_assessment_inputs_still_present():
    assert (ROOT / "data" / "validation.csv").is_file()
    assert (ROOT / "data" / "december-chart-inputs.csv").is_file()
    assert (ROOT / "score.py").is_file()


def test_secrets_are_ignored_by_git():
    ignore = (ROOT / ".gitignore").read_text()
    for pattern in (".env", "*.key", "*.pem", "secrets.json"):
        assert pattern in ignore.split(), pattern
    assert (ROOT / ".env.example").is_file()
