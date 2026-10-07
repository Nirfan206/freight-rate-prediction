from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(ROOT / "src")
)

from data_cleaning import (
    fit_cleaning_statistics,
    apply_cleaning
)

from features import (
    build_features,
    get_model_features
)


def test_training_file_exists():

    path = ROOT / "data" / "train-test.csv"

    assert path.exists()


def test_training_shape():

    path = ROOT / "data" / "train-test.csv"

    df = pd.read_csv(path)

    assert len(df) == 48000
    assert len(df.columns) == 14


def test_validation_shape():

    path = ROOT / "data" / "validation.csv"

    df = pd.read_csv(path)

    assert len(df) == 12000


def test_required_training_columns():

    path = ROOT / "data" / "train-test.csv"

    df = pd.read_csv(path)

    required = [
        "load_id",
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
        "posted_rate"
    ]

    for column in required:

        assert column in df.columns


def test_cleaning_pipeline():

    path = ROOT / "data" / "train-test.csv"

    df = pd.read_csv(path)

    stats = fit_cleaning_statistics(
        df
    )

    cleaned = apply_cleaning(
        df,
        stats
    )

    assert len(cleaned) == len(df)

    assert "weight_missing" in cleaned.columns

    assert "market_index_missing" in cleaned.columns


def test_feature_pipeline():

    path = ROOT / "data" / "train-test.csv"

    df = pd.read_csv(path)

    stats = fit_cleaning_statistics(
        df
    )

    cleaned = apply_cleaning(
        df,
        stats
    )

    features = build_features(
        cleaned
    )

    model_features = get_model_features(
        features
    )

    assert len(model_features) == 43

    assert "posted_rate" not in model_features

    assert "load_id" not in model_features


def test_model_artifact_exists():

    path = (
        ROOT /
        "artifacts" /
        "ridge_model.joblib"
    )

    assert path.exists()


def test_model_artifact():

    path = (
        ROOT /
        "artifacts" /
        "ridge_model.joblib"
    )

    artifact = joblib.load(path)

    assert "model" in artifact
    assert "feature_columns" in artifact
    assert "cleaning_stats" in artifact

    assert len(
        artifact["feature_columns"]
    ) == 43


def test_validation_predictions():

    path = (
        ROOT /
        "validation_predictions.csv"
    )

    assert path.exists()

    df = pd.read_csv(path)

    assert df.shape == (12000, 2)

    assert df.columns.tolist() == [
        "load_id",
        "predicted_rate"
    ]

    assert df["load_id"].is_unique

    assert not df["predicted_rate"].isna().any()

    assert np.isfinite(
        df["predicted_rate"]
    ).all()

    assert (
        df["predicted_rate"] > 0
    ).all()


def test_validation_ids():

    path = (
        ROOT /
        "validation_predictions.csv"
    )

    df = pd.read_csv(path)

    expected_ids = [
        f"TE-{i:06d}"
        for i in range(1, 12001)
    ]

    assert (
        df["load_id"].tolist()
        == expected_ids
    )


def test_december_predictions():

    path = (
        ROOT /
        "outputs" /
        "predictions" /
        "december_predictions.csv"
    )

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 31

    assert df.columns.tolist() == [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate"
    ]

    assert not df["predicted_rate"].isna().any()

    assert np.isfinite(
        df["predicted_rate"]
    ).all()

    assert (
        df["predicted_rate"] > 0
    ).all()


def test_december_chart_exists():

    path = (
        ROOT /
        "scorer_results" /
        "candidate_december.png"
    )

    assert path.exists()