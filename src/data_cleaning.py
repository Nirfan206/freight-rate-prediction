"""
Data cleaning and quality handling for the Spotter Freight Rate
Prediction project.

Important design principle:
Cleaning decisions must not leak information from validation
periods into training periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

WEIGHT_CAP = 47_500.0


# ============================================================
# Cleaning statistics
# ============================================================

@dataclass
class CleaningStatistics:
    """
    Statistics learned from a training dataset.

    These statistics are fitted only on the data available to the
    model at training time.
    """

    weight_median: float
    market_index_global_median: float
    market_index_by_date: dict[str, float]


# ============================================================
# Fitting cleaning statistics
# ============================================================

def fit_cleaning_statistics(
    df: pd.DataFrame,
) -> CleaningStatistics:
    """
    Learn imputation statistics from a training dataframe.

    This function must ONLY be called on the training portion of
    a temporal fold, never on the complete dataset before splitting.
    """

    if "weight" not in df.columns:
        raise ValueError("Missing required column: weight")

    if "market_index" not in df.columns:
        raise ValueError(
            "Missing required column: market_index"
        )

    working = df.copy()

    working["date"] = pd.to_datetime(
        working["date"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Weight median
    # --------------------------------------------------------

    weight_median = working["weight"].median()

    if pd.isna(weight_median):
        raise ValueError(
            "Unable to calculate training weight median."
        )

    # --------------------------------------------------------
    # Market index global fallback
    # --------------------------------------------------------

    market_index_global_median = (
        working["market_index"].median()
    )

    if pd.isna(market_index_global_median):
        raise ValueError(
            "Unable to calculate market_index median."
        )

    # --------------------------------------------------------
    # Market index by-date statistics
    # --------------------------------------------------------

    market_by_date = (
        working
        .dropna(subset=["date"])
        .groupby("date")["market_index"]
        .mean()
    )

    market_index_by_date = {
        date.strftime("%Y-%m-%d"): float(value)
        for date, value in market_by_date.items()
        if pd.notna(value)
    }

    return CleaningStatistics(
        weight_median=float(weight_median),
        market_index_global_median=float(
            market_index_global_median
        ),
        market_index_by_date=market_index_by_date,
    )


# ============================================================
# Apply cleaning
# ============================================================

def apply_cleaning(
    df: pd.DataFrame,
    statistics: CleaningStatistics,
) -> pd.DataFrame:
    """
    Apply cleaning using statistics previously learned from
    the training data.

    No statistics are calculated from the dataframe being
    transformed.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if "date" not in result.columns:
        raise ValueError(
            "Missing required column: date"
        )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    if result["date"].isna().any():
        invalid_count = int(
            result["date"].isna().sum()
        )

        raise ValueError(
            f"Found {invalid_count} invalid date values."
        )

    # --------------------------------------------------------
    # Distance validation
    # --------------------------------------------------------

    if "distance" not in result.columns:
        raise ValueError(
            "Missing required column: distance"
        )

    invalid_distance = result["distance"] <= 0

    result["distance_invalid"] = (
        invalid_distance.astype("int8")
    )

    if invalid_distance.any():
        raise ValueError(
            "Non-positive distance values found. "
            "These must be investigated before modeling."
        )

    # --------------------------------------------------------
    # Weight
    # --------------------------------------------------------

    if "weight" not in result.columns:
        raise ValueError(
            "Missing required column: weight"
        )

    result["weight_missing"] = (
        result["weight"]
        .isna()
        .astype("int8")
    )

    result["weight_negative"] = (
        result["weight"]
        < 0
    ).astype("int8")

    result["weight_at_positive_cap"] = (
        result["weight"]
        >= WEIGHT_CAP
    ).astype("int8")

    result["weight_at_negative_cap"] = (
        result["weight"]
        <= -WEIGHT_CAP
    ).astype("int8")

    # Preserve original value.
    result["weight_raw"] = result["weight"]

    # Median imputation.
    result["weight"] = result["weight"].fillna(
        statistics.weight_median
    )

    # Absolute-weight candidate.
    #
    # IMPORTANT:
    # We are NOT replacing the main weight with abs(weight).
    # This candidate is kept for later experiments.
    result["weight_absolute"] = (
        result["weight"].abs()
    )

    # --------------------------------------------------------
    # Market index
    # --------------------------------------------------------

    if "market_index" not in result.columns:
        raise ValueError(
            "Missing required column: market_index"
        )

    result["market_index_missing"] = (
        result["market_index"]
        .isna()
        .astype("int8")
    )

    # Date-specific market index fallback.
    date_keys = result["date"].dt.strftime(
        "%Y-%m-%d"
    )

    date_values = date_keys.map(
        statistics.market_index_by_date
    )

    result["market_index"] = (
        result["market_index"]
        .fillna(date_values)
        .fillna(
            statistics.market_index_global_median
        )
    )

    # --------------------------------------------------------
    # Numeric safety
    # --------------------------------------------------------

    numeric_columns = [
        "distance",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "weight",
        "weight_raw",
        "weight_absolute",
        "market_index",
        "quote_signal",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Target is NOT modified here.
    # --------------------------------------------------------

    return result


# ============================================================
# Basic cleaning pipeline
# ============================================================

def clean_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, CleaningStatistics]:
    """
    Fit cleaning statistics and transform training data.

    This is intended for final training on the complete labeled
    dataset.
    """

    statistics = fit_cleaning_statistics(df)

    cleaned = apply_cleaning(
        df,
        statistics,
    )

    return cleaned, statistics


# ============================================================
# Diagnostic report
# ============================================================

def create_cleaning_report(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a before/after diagnostic summary.
    """

    rows = []

    # Weight
    if "weight" in original.columns:
        rows.append({
            "issue": "missing_weight",
            "original_count": int(
                original["weight"].isna().sum()
            ),
            "cleaned_count": int(
                cleaned["weight"].isna().sum()
            ),
            "action": (
                "Median imputation using training "
                "statistics; missingness flag retained."
            ),
        })

        rows.append({
            "issue": "negative_weight",
            "original_count": int(
                (original["weight"] < 0).sum()
            ),
            "cleaned_count": int(
                cleaned["weight_negative"].sum()
            ),
            "action": (
                "Preserved raw value and created "
                "anomaly flag; no automatic deletion."
            ),
        })

    # Market index
    if "market_index" in original.columns:
        rows.append({
            "issue": "missing_market_index",
            "original_count": int(
                original["market_index"].isna().sum()
            ),
            "cleaned_count": int(
                cleaned["market_index"].isna().sum()
            ),
            "action": (
                "Date-aware training-period mean with "
                "global training fallback."
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Main diagnostic execution
# ============================================================

def main() -> None:
    """
    Run a diagnostic cleaning operation on the training data.

    This does not train a model.
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
        / "data_audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DATA QUALITY / CLEANING DIAGNOSTIC")
    print("=" * 70)

    train = pd.read_csv(train_file)

    print(
        f"\nOriginal shape: {train.shape}"
    )

    cleaned, statistics = clean_training_data(
        train
    )

    print(
        f"Cleaned shape:  {cleaned.shape}"
    )

    print(
        "\nTraining weight median used:",
        statistics.weight_median,
    )

    print(
        "Training market index median:",
        statistics.market_index_global_median,
    )

    print(
        "\nRemaining missing values:"
    )

    print(
        cleaned[
            [
                "weight",
                "market_index",
            ]
        ]
        .isna()
        .sum()
    )

    report = create_cleaning_report(
        train,
        cleaned,
    )

    report_path = (
        output_dir
        / "cleaning_diagnostic.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print(
        "\nDiagnostic report saved to:"
    )

    print(report_path)

    print(
        "\nCleaning diagnostic completed."
    )


if __name__ == "__main__":
    main()