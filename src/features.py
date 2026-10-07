"""
Feature engineering for the Spotter Freight Rate Prediction project.

All features generated here must be available at prediction time.
No target-derived feature is created in this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

EARTH_RADIUS_MILES = 3958.7613


# ============================================================
# Haversine distance
# ============================================================

def haversine_distance(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Calculate great-circle distance in miles between two
    geographic coordinates.
    """

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_MILES * c


# ============================================================
# Base features
# ============================================================

def create_base_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create deterministic numerical and date-based features.

    Assumes data has already passed through the cleaning layer.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    if result["date"].isna().any():
        raise ValueError(
            "Invalid date found during feature engineering."
        )

    result["day_of_week"] = (
        result["date"].dt.dayofweek
    )

    result["day_of_month"] = (
        result["date"].dt.day
    )

    result["day_of_year"] = (
        result["date"].dt.dayofyear
    )

    # --------------------------------------------------------
    # Distance transformations
    # --------------------------------------------------------

    if "distance" not in result.columns:
        raise ValueError(
            "Missing required feature: distance"
        )

    if (result["distance"] <= 0).any():
        raise ValueError(
            "Distance must be positive."
        )

    result["log_distance"] = np.log1p(
        result["distance"]
    )

    # --------------------------------------------------------
    # Weight transformations
    # --------------------------------------------------------

    if "weight" in result.columns:
        result["log_weight"] = np.log1p(
            result["weight"].clip(lower=0)
        )

        if "weight_absolute" not in result.columns:
            result["weight_absolute"] = (
                result["weight"].abs()
            )

        result["weight_distance_ratio"] = (
            result["weight_absolute"]
            / result["distance"]
        )

    # --------------------------------------------------------
    # Geographic features
    # --------------------------------------------------------

    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]

    if all(
        column in result.columns
        for column in coordinate_columns
    ):

        result["latitude_difference"] = (
            result["delivery_lat"]
            - result["pickup_lat"]
        )

        result["longitude_difference"] = (
            result["delivery_lon"]
            - result["pickup_lon"]
        )

        result["latitude_difference_abs"] = (
            result["latitude_difference"].abs()
        )

        result["longitude_difference_abs"] = (
            result["longitude_difference"].abs()
        )

        result["haversine_distance"] = (
            haversine_distance(
                result["pickup_lat"],
                result["pickup_lon"],
                result["delivery_lat"],
                result["delivery_lon"],
            )
        )

        result["distance_haversine_ratio"] = (
            result["distance"]
            / result["haversine_distance"].replace(
                0,
                np.nan,
            )
        )

        result["distance_haversine_difference"] = (
            result["distance"]
            - result["haversine_distance"]
        )

    # --------------------------------------------------------
    # Freight-specific interactions
    # --------------------------------------------------------

    if "market_index" in result.columns:

        result["market_distance_interaction"] = (
            result["market_index"]
            * result["log_distance"]
        )

    if "weight" in result.columns:

        result["weight_distance_interaction"] = (
            result["weight_absolute"]
            * result["log_distance"]
        )

    # Equipment-specific numeric interactions are created
    # after one-hot encoding if required.

    return result


# ============================================================
# One-hot categorical features
# ============================================================

def encode_equipment(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One-hot encode equipment.

    The function uses get_dummies with a stable category list
    so train and inference can be aligned.
    """

    result = df.copy()

    if "equipment" not in result.columns:
        raise ValueError(
            "Missing required feature: equipment"
        )

    equipment_categories = [
        "Dry Van",
        "Reefer",
        "Flatbed",
    ]

    result["equipment"] = (
        result["equipment"]
        .fillna("Unknown")
        .astype(str)
    )

    for equipment in equipment_categories:
        column_name = (
            "equipment_"
            + equipment.lower().replace(" ", "_")
        )

        result[column_name] = (
            result["equipment"] == equipment
        ).astype("int8")

    result["equipment_unknown"] = (
        ~result["equipment"].isin(
            equipment_categories
        )
    ).astype("int8")

    return result


# ============================================================
# Equipment interactions
# ============================================================

def create_equipment_interactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create interactions between distance and equipment.
    """

    result = df.copy()

    equipment_columns = [
        column
        for column in result.columns
        if column.startswith("equipment_")
    ]

    for column in equipment_columns:

        result[
            f"{column}_log_distance"
        ] = (
            result[column]
            * result["log_distance"]
        )

        if "market_index" in result.columns:

            result[
                f"{column}_market_index"
            ] = (
                result[column]
                * result["market_index"]
            )

    return result


# ============================================================
# Complete feature builder
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complete deterministic feature-engineering pipeline.

    No target-derived features are created.
    """

    result = create_base_features(df)

    result = encode_equipment(result)

    result = create_equipment_interactions(result)

    return result


# ============================================================
# Feature selection
# ============================================================

def get_model_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return model-safe feature columns.

    Explicitly excludes:
    - load_id
    - posted_rate
    - rate_per_mile
    - raw date
    - city text fields

    Geographic coordinates remain available because they
    generalize to unseen city names.
    """

    excluded = {
        "load_id",
        "posted_rate",
        "rate_per_mile",
        "pickup",
        "delivery",
        "equipment",
        "date",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    # Keep only numeric columns.
    features = [
        column
        for column in features
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    return features


# ============================================================
# Diagnostics
# ============================================================

def feature_diagnostics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a diagnostic table for generated features.
    """

    rows = []

    for column in df.columns:

        rows.append({
            "feature": column,
            "dtype": str(df[column].dtype),
            "missing": int(
                df[column].isna().sum()
            ),
            "unique_values": int(
                df[column].nunique(
                    dropna=False
                )
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Main diagnostic
# ============================================================

def main() -> None:
    """
    Build features on the cleaned training dataset and save
    feature diagnostics.
    """

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    train_file = (
        project_root
        / "data"
        / "train-test.csv"
    )

    output_dir = (
        project_root
        / "outputs"
        / "experiments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("FEATURE ENGINEERING DIAGNOSTIC")
    print("=" * 70)

    train = pd.read_csv(train_file)

    train["date"] = pd.to_datetime(
        train["date"]
    )

    # Import cleaning layer.
    from data_cleaning import (
        clean_training_data,
    )

    cleaned, _ = clean_training_data(
        train
    )

    print(
        f"\nOriginal columns: {len(train.columns)}"
    )

    features = build_features(cleaned)

    print(
        f"Feature dataframe columns: "
        f"{len(features.columns)}"
    )

    model_features = get_model_features(
        features
    )

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    print("\nModel feature list:")

    for index, feature in enumerate(
        model_features,
        start=1,
    ):
        print(
            f"{index:02d}. {feature}"
        )

    diagnostics = feature_diagnostics(
        features
    )

    diagnostics.to_csv(
        output_dir
        / "feature_diagnostics.csv",
        index=False,
    )

    pd.DataFrame({
        "feature": model_features
    }).to_csv(
        output_dir
        / "model_feature_list.csv",
        index=False,
    )

    print(
        "\nFeature diagnostics saved to:"
    )

    print(output_dir)

    print(
        "\nFeature engineering diagnostic completed."
    )


if __name__ == "__main__":
    main()