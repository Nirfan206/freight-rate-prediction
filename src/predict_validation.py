from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs" / "predictions"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.append(
    str(ROOT / "src")
)

from data_cleaning import apply_cleaning
from features import build_features


MODEL_FILE = (
    ARTIFACT_DIR /
    "ridge_model.joblib"
)

VALIDATION_FILE = (
    DATA_DIR /
    "validation.csv"
)

TEMPLATE_FILE = (
    DATA_DIR /
    "validation-predictions-template.csv"
)

OUTPUT_FILE = (
    ROOT /
    "validation_predictions.csv"
)


def main():

    print("=" * 75)
    print("PHASE 10B - VALIDATION PREDICTIONS")
    print("=" * 75)

    # ---------------------------------------------------------
    # CHECK MODEL
    # ---------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_FILE}\n"
            "Run: python src\\final_pipeline.py"
        )

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    cleaning_stats = artifact["cleaning_stats"]

    print(
        f"\nLoaded model: {MODEL_FILE}"
    )

    print(
        f"Expected features: "
        f"{len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # LOAD VALIDATION DATA
    # ---------------------------------------------------------

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"\nValidation rows: "
        f"{len(validation):,}"
    )

    if len(validation) != 12000:

        raise ValueError(
            "Validation dataset must contain "
            "exactly 12,000 rows."
        )

    # ---------------------------------------------------------
    # CLEAN
    # ---------------------------------------------------------

    validation_clean = apply_cleaning(
        validation,
        cleaning_stats
    )

    # ---------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------

    validation_features = build_features(
        validation_clean
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in validation_features.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required model features:\n"
            + "\n".join(missing_features)
        )

    X_validation = validation_features[
        feature_columns
    ].copy()

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    log_predictions = model.predict(
        X_validation
    )

    predictions = np.expm1(
        log_predictions
    )

    # Rates must be positive.
    predictions = np.maximum(
        predictions,
        1.0
    )

    # ---------------------------------------------------------
    # BUILD REQUIRED OUTPUT
    # ---------------------------------------------------------

    result = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": predictions
    })

    # ---------------------------------------------------------
    # VALIDATE IDs
    # ---------------------------------------------------------

    expected_ids = [
        f"TE-{i:06d}"
        for i in range(
            1,
            12001
        )
    ]

    actual_ids = result[
        "load_id"
    ].astype(str).tolist()

    if actual_ids != expected_ids:

        raise ValueError(
            "Validation load IDs do not match "
            "the expected TE-000001 ... TE-012000 sequence."
        )

    # ---------------------------------------------------------
    # VALIDATE PREDICTIONS
    # ---------------------------------------------------------

    if result["predicted_rate"].isna().any():

        raise ValueError(
            "Predictions contain NaN values."
        )

    if not np.isfinite(
        result["predicted_rate"]
    ).all():

        raise ValueError(
            "Predictions contain non-finite values."
        )

    if (
        result["predicted_rate"] <= 0
    ).any():

        raise ValueError(
            "Predictions must be positive."
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\nPrediction summary:")

    print(
        result["predicted_rate"]
        .describe()
        .to_string()
    )

    print("\nFirst 10 predictions:")

    print(
        result.head(10).to_string(
            index=False
        )
    )

    print("\nLast 10 predictions:")

    print(
        result.tail(10).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # TEMPLATE CHECK
    # ---------------------------------------------------------

    if TEMPLATE_FILE.exists():

        template = pd.read_csv(
            TEMPLATE_FILE
        )

        print(
            f"\nTemplate rows: "
            f"{len(template):,}"
        )

        if len(template) != len(result):

            raise ValueError(
                "Template and prediction row counts differ."
            )

        print(
            "Template row count matches."
        )

    print("\n" + "=" * 75)
    print("VALIDATION PREDICTIONS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()