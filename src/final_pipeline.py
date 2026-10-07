from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs" / "predictions"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))

from data_cleaning import (
    fit_cleaning_statistics,
    apply_cleaning,
)

from features import (
    build_features,
    get_model_features,
)


MODEL_FILE = ARTIFACT_DIR / "ridge_model.joblib"
METADATA_FILE = ARTIFACT_DIR / "model_metadata.json"


def train_model():

    print("=" * 75)
    print("PHASE 10A - FINAL MODEL TRAINING")
    print("=" * 75)

    train_path = DATA_DIR / "train-test.csv"

    df = pd.read_csv(train_path)

    print(
        f"\nDevelopment data: {df.shape}"
    )

    # ---------------------------------------------------------
    # CLEANING
    # ---------------------------------------------------------

    cleaning_stats = fit_cleaning_statistics(df)

    clean_df = apply_cleaning(
        df,
        cleaning_stats
    )

    # ---------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------

    feature_df = build_features(
        clean_df
    )

    feature_columns = get_model_features(
        feature_df
    )

    X = feature_df[
        feature_columns
    ].copy()

    y = feature_df[
        "posted_rate"
    ].copy()

    print(
        f"Rows: {len(X):,}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(
                alpha=0.01
            )
        )
    ])

    print(
        "\nTraining Ridge(alpha=0.01) "
        "using log-transformed target..."
    )

    model.fit(
        X,
        np.log1p(y)
    )

    # ---------------------------------------------------------
    # SAVE ARTIFACT
    # ---------------------------------------------------------

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "cleaning_stats": cleaning_stats,
    }

    joblib.dump(
        artifact,
        MODEL_FILE
    )

    metadata = {
        "model": "Ridge",
        "alpha": 0.01,
        "target_transform": "log1p",
        "inverse_transform": "expm1",
        "training_rows": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "features": feature_columns,
        "training_start": str(
            df["date"].min()
        ),
        "training_end": str(
            df["date"].max()
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"\nModel saved:\n{MODEL_FILE}"
    )

    print(
        f"\nMetadata saved:\n{METADATA_FILE}"
    )

    print("\nFinal model training complete.")


if __name__ == "__main__":
    train_model()