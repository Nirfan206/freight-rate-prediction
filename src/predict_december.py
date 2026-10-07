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


MODEL_FILE = ARTIFACT_DIR / "ridge_model.joblib"

DECEMBER_FILE = (
    DATA_DIR / "december-chart-inputs.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "december_predictions.csv"
)


def find_city_coordinates(train, city):

    pickup_match = train[
        train["pickup"] == city
    ]

    if len(pickup_match) > 0:

        row = pickup_match.iloc[0]

        return (
            row["pickup_lat"],
            row["pickup_lon"]
        )

    delivery_match = train[
        train["delivery"] == city
    ]

    if len(delivery_match) > 0:

        row = delivery_match.iloc[0]

        return (
            row["delivery_lat"],
            row["delivery_lon"]
        )

    return (
        np.nan,
        np.nan
    )


def main():

    print("=" * 75)
    print("PHASE 10C - DECEMBER PREDICTIONS")
    print("=" * 75)

    # ---------------------------------------------------------
    # CHECK MODEL
    # ---------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Final model artifact not found.\n"
            "Run:\n"
            "python src\\final_pipeline.py"
        )

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]

    feature_columns = artifact[
        "feature_columns"
    ]

    cleaning_stats = artifact[
        "cleaning_stats"
    ]

    print(
        f"\nLoaded model:\n{MODEL_FILE}"
    )

    print(
        f"Model features: {len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # LOAD DEVELOPMENT DATA
    # ---------------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train-test.csv"
    )

    print(
        f"Development rows: {len(train):,}"
    )

    # ---------------------------------------------------------
    # LOAD DECEMBER INPUT
    # ---------------------------------------------------------

    december = pd.read_csv(
        DECEMBER_FILE
    )

    print(
        f"\nDecember rows: {len(december)}"
    )

    if len(december) != 31:

        raise ValueError(
            "December input must contain exactly 31 rows."
        )

    # ---------------------------------------------------------
    # CHECK DECEMBER DATES
    # ---------------------------------------------------------

    december_dates = pd.to_datetime(
        december["date"]
    )

    if december_dates.dt.month.nunique() != 1:

        raise ValueError(
            "December input must contain only December dates."
        )

    if december_dates.dt.day.nunique() != 31:

        raise ValueError(
            "December input must contain all 31 days."
        )

    # ---------------------------------------------------------
    # ADD COORDINATES
    # ---------------------------------------------------------

    december["pickup_lat"] = np.nan
    december["pickup_lon"] = np.nan

    december["delivery_lat"] = np.nan
    december["delivery_lon"] = np.nan

    for index, row in december.iterrows():

        pickup_lat, pickup_lon = (
            find_city_coordinates(
                train,
                row["pickup"]
            )
        )

        delivery_lat, delivery_lon = (
            find_city_coordinates(
                train,
                row["delivery"]
            )
        )

        december.loc[
            index,
            "pickup_lat"
        ] = pickup_lat

        december.loc[
            index,
            "pickup_lon"
        ] = pickup_lon

        december.loc[
            index,
            "delivery_lat"
        ] = delivery_lat

        december.loc[
            index,
            "delivery_lon"
        ] = delivery_lon

    # ---------------------------------------------------------
    # CHECK COORDINATES
    # ---------------------------------------------------------

    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon"
    ]

    missing_coordinates = (
        december[
            coordinate_columns
        ]
        .isna()
        .sum()
    )

    print(
        "\nMissing coordinates:"
    )

    print(
        missing_coordinates.to_string()
    )

    # ---------------------------------------------------------
    # ADD MISSING MODEL INPUT COLUMNS
    # ---------------------------------------------------------

    # The December assessment input does not contain
    # market_index or quote_signal.
    #
    # We therefore use training-data median fallback values.

    market_index_median = (
        train["market_index"]
        .median()
    )

    quote_signal_median = (
        train["quote_signal"]
        .median()
    )

    december["market_index"] = (
        market_index_median
    )

    december["quote_signal"] = (
        quote_signal_median
    )

    print(
        f"\nMarket index fallback: "
        f"{market_index_median:.6f}"
    )

    print(
        f"Quote signal fallback: "
        f"{quote_signal_median:.6f}"
    )

    # ---------------------------------------------------------
    # CLEANING
    # ---------------------------------------------------------

    december_clean = apply_cleaning(
        december,
        cleaning_stats
    )

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------

    december_features = build_features(
        december_clean
    )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in december_features.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features:\n"
            + "\n".join(
                missing_features
            )
        )

    X_december = december_features[
        feature_columns
    ].copy()

    print(
        f"\nDecember feature matrix: "
        f"{X_december.shape}"
    )

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    print(
        "\nGenerating December predictions..."
    )

    log_prediction = model.predict(
        X_december
    )

    predictions = np.expm1(
        log_prediction
    )

    predictions = np.maximum(
        predictions,
        1.0
    )

    # ---------------------------------------------------------
    # BUILD OFFICIAL DECEMBER OUTPUT
    # ---------------------------------------------------------

    result = december[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date"
        ]
    ].copy()

    result["predicted_rate"] = predictions

    result = result[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate"
        ]
    ]

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    required_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate"
    ]

    if result.columns.tolist() != required_columns:

        raise ValueError(
            "December output columns are incorrect."
        )

    if len(result) != 31:

        raise ValueError(
            "December output must contain exactly 31 rows."
        )

    if result["predicted_rate"].isna().any():

        raise ValueError(
            "December predictions contain NaN values."
        )

    if not np.isfinite(
        result["predicted_rate"]
    ).all():

        raise ValueError(
            "December predictions contain "
            "non-finite values."
        )

    if (
        result["predicted_rate"] <= 0
    ).any():

        raise ValueError(
            "December predictions must be positive."
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

    print(
        "\nDecember prediction summary:"
    )

    print(
        result["predicted_rate"]
        .describe()
        .to_string()
    )

    print(
        "\nPredictions:"
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "DECEMBER PREDICTIONS COMPLETE"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()