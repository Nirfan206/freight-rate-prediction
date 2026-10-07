"""
Comprehensive data audit for the Spotter Freight Rate Prediction assessment.

This module inspects the supplied datasets and writes reproducible
audit outputs under outputs/data_audit/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_audit"

TRAIN_FILE = DATA_DIR / "train-test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEMPLATE_FILE = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_FILE = DATA_DIR / "december-chart-inputs.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def safe_json_value(value):
    """Convert pandas/numpy values into JSON-compatible values."""

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def dataframe_basic_info(df: pd.DataFrame) -> dict:
    """Return basic information about a dataframe."""

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2),
            3,
        ),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing-value summary."""

    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
    })

    result["missing_percentage"] = (
        result["missing_count"] / len(df) * 100
    ).round(4)

    result["dtype"] = [
        str(df[column].dtype)
        for column in df.columns
    ]

    return result


def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create numerical descriptive statistics."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) == 0:
        return pd.DataFrame()

    summary = (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "column"})
    )

    return summary


def categorical_summary(
    df: pd.DataFrame,
    max_values: int = 100,
) -> dict:
    """Return value counts for categorical columns."""

    result = {}

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        counts = df[column].value_counts(
            dropna=False
        ).head(max_values)

        result[column] = {
            str(key): int(value)
            for key, value in counts.items()
        }

    return result


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with date parsed when available."""

    result = df.copy()

    if "date" in result.columns:
        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce",
        )

    return result


# ============================================================
# Dataset audit
# ============================================================

def audit_dataset(
    df: pd.DataFrame,
    name: str,
) -> dict:
    """Create a detailed audit dictionary."""

    df = parse_dates(df)

    result = {
        "name": name,
        "basic_info": dataframe_basic_info(df),
        "missing_values": {},
        "numeric_summary": {},
        "categorical_summary": categorical_summary(df),
    }

    # Missing values
    for column in df.columns:
        missing = int(df[column].isna().sum())

        result["missing_values"][column] = {
            "count": missing,
            "percentage": round(
                missing / len(df) * 100,
                4,
            ),
        }

    # Numeric summary
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        series = df[column]

        result["numeric_summary"][column] = {
            "count": int(series.count()),
            "mean": safe_json_value(series.mean()),
            "std": safe_json_value(series.std()),
            "min": safe_json_value(series.min()),
            "25_percentile": safe_json_value(
                series.quantile(0.25)
            ),
            "median": safe_json_value(
                series.median()
            ),
            "75_percentile": safe_json_value(
                series.quantile(0.75)
            ),
            "max": safe_json_value(series.max()),
        }

    # Date range
    if "date" in df.columns:
        valid_dates = df["date"].dropna()

        if not valid_dates.empty:
            result["date_range"] = {
                "min": valid_dates.min().strftime("%Y-%m-%d"),
                "max": valid_dates.max().strftime("%Y-%m-%d"),
                "invalid_dates": int(
                    df["date"].isna().sum()
                ),
            }

    # ID checks
    if "load_id" in df.columns:
        result["load_id"] = {
            "unique_count": int(df["load_id"].nunique()),
            "duplicate_count": int(
                df["load_id"].duplicated().sum()
            ),
            "missing_count": int(
                df["load_id"].isna().sum()
            ),
        }

    return result


# ============================================================
# Freight-specific checks
# ============================================================

def freight_checks(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    template: pd.DataFrame,
    december: pd.DataFrame,
) -> dict:
    """Run freight-specific consistency checks."""

    train = parse_dates(train)
    validation = parse_dates(validation)
    december = parse_dates(december)

    result = {}

    # --------------------------------------------------------
    # Weight checks
    # --------------------------------------------------------

    if "weight" in train.columns:
        result["train_negative_weight"] = int(
            (train["weight"] < 0).sum()
        )

        result["train_zero_weight"] = int(
            (train["weight"] == 0).sum()
        )

        result["train_weight_max"] = safe_json_value(
            train["weight"].max()
        )

    if "weight" in validation.columns:
        result["validation_negative_weight"] = int(
            (validation["weight"] < 0).sum()
        )

        result["validation_zero_weight"] = int(
            (validation["weight"] == 0).sum()
        )

        result["validation_weight_max"] = safe_json_value(
            validation["weight"].max()
        )

    # --------------------------------------------------------
    # Distance checks
    # --------------------------------------------------------

    if "distance" in train.columns:
        result["train_non_positive_distance"] = int(
            (train["distance"] <= 0).sum()
        )

    if "distance" in validation.columns:
        result["validation_non_positive_distance"] = int(
            (validation["distance"] <= 0).sum()
        )

    # --------------------------------------------------------
    # Target checks
    # --------------------------------------------------------

    if "posted_rate" in train.columns:
        target = train["posted_rate"].dropna()

        result["target"] = {
            "count": int(target.count()),
            "min": safe_json_value(target.min()),
            "median": safe_json_value(target.median()),
            "mean": safe_json_value(target.mean()),
            "max": safe_json_value(target.max()),
            "negative_count": int((target < 0).sum()),
            "zero_count": int((target == 0).sum()),
            "greater_than_5000": int((target > 5000).sum()),
            "less_than_1000": int((target < 1000).sum()),
        }

        if "distance" in train.columns:
            valid = train[
                (train["distance"] > 0)
                & train["posted_rate"].notna()
            ].copy()

            valid["rate_per_mile"] = (
                valid["posted_rate"]
                / valid["distance"]
            )

            result["rate_per_mile"] = {
                "min": safe_json_value(
                    valid["rate_per_mile"].min()
                ),
                "median": safe_json_value(
                    valid["rate_per_mile"].median()
                ),
                "mean": safe_json_value(
                    valid["rate_per_mile"].mean()
                ),
                "max": safe_json_value(
                    valid["rate_per_mile"].max()
                ),
                "greater_than_5": int(
                    (valid["rate_per_mile"] > 5).sum()
                ),
                "less_than_1": int(
                    (valid["rate_per_mile"] < 1).sum()
                ),
            }

    # --------------------------------------------------------
    # Equipment
    # --------------------------------------------------------

    for name, dataframe in [
        ("train", train),
        ("validation", validation),
        ("december", december),
    ]:
        if "equipment" in dataframe.columns:
            result[f"{name}_equipment"] = {
                str(key): int(value)
                for key, value in dataframe[
                    "equipment"
                ].value_counts(dropna=False).items()
            }

    # --------------------------------------------------------
    # City checks
    # --------------------------------------------------------

    train_pickup = set()

    train_delivery = set()

    validation_pickup = set()

    validation_delivery = set()

    if "pickup" in train.columns:
        train_pickup = set(
            train["pickup"]
            .dropna()
            .astype(str)
            .unique()
        )

    if "delivery" in train.columns:
        train_delivery = set(
            train["delivery"]
            .dropna()
            .astype(str)
            .unique()
        )

    if "pickup" in validation.columns:
        validation_pickup = set(
            validation["pickup"]
            .dropna()
            .astype(str)
            .unique()
        )

    if "delivery" in validation.columns:
        validation_delivery = set(
            validation["delivery"]
            .dropna()
            .astype(str)
            .unique()
        )

    unseen_pickup = sorted(
        validation_pickup - train_pickup
    )

    unseen_delivery = sorted(
        validation_delivery - train_delivery
    )

    result["cities"] = {
        "train_pickup_count": len(train_pickup),
        "train_delivery_count": len(train_delivery),
        "validation_pickup_count": len(validation_pickup),
        "validation_delivery_count": len(validation_delivery),
        "unseen_pickup_cities": unseen_pickup,
        "unseen_delivery_cities": unseen_delivery,
    }

    # --------------------------------------------------------
    # Route overlap
    # --------------------------------------------------------

    if (
        "pickup" in train.columns
        and "delivery" in train.columns
        and "pickup" in validation.columns
        and "delivery" in validation.columns
    ):
        train_routes = set(
            zip(
                train["pickup"].astype(str),
                train["delivery"].astype(str),
            )
        )

        validation_routes = list(
            zip(
                validation["pickup"].astype(str),
                validation["delivery"].astype(str),
            )
        )

        known_route_count = sum(
            route in train_routes
            for route in validation_routes
        )

        total_validation_routes = len(
            validation_routes
        )

        result["routes"] = {
            "train_unique_routes": len(train_routes),
            "validation_unique_routes": int(
                validation[
                    ["pickup", "delivery"]
                ].drop_duplicates().shape[0]
            ),
            "validation_rows_on_known_routes": int(
                known_route_count
            ),
            "validation_rows_on_known_routes_percentage": round(
                known_route_count
                / total_validation_routes
                * 100,
                4,
            ),
        }

    # --------------------------------------------------------
    # Template ID alignment
    # --------------------------------------------------------

    if (
        "load_id" in validation.columns
        and "load_id" in template.columns
    ):
        validation_ids = (
            validation["load_id"]
            .astype(str)
            .tolist()
        )

        template_ids = (
            template["load_id"]
            .astype(str)
            .tolist()
        )

        result["template_validation_alignment"] = {
            "same_row_count": len(validation_ids)
            == len(template_ids),
            "same_ids": set(validation_ids)
            == set(template_ids),
            "same_order": validation_ids
            == template_ids,
        }

    # --------------------------------------------------------
    # December checks
    # --------------------------------------------------------

    if "date" in december.columns:
        valid_dates = december["date"].dropna()

        result["december"] = {
            "row_count": int(len(december)),
            "unique_dates": int(
                valid_dates.nunique()
            ),
            "min_date": (
                valid_dates.min().strftime("%Y-%m-%d")
                if not valid_dates.empty
                else None
            ),
            "max_date": (
                valid_dates.max().strftime("%Y-%m-%d")
                if not valid_dates.empty
                else None
            ),
        }

    return result


# ============================================================
# Main audit
# ============================================================

def main() -> None:
    """Run the complete audit."""

    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION — DATA AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    template = pd.read_csv(TEMPLATE_FILE)
    december = pd.read_csv(DECEMBER_FILE)

    print(f"Train:      {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Template:   {template.shape}")
    print(f"December:   {december.shape}")

    # --------------------------------------------------------
    # Basic audits
    # --------------------------------------------------------

    audit = {
        "train": audit_dataset(
            train,
            "train-test.csv",
        ),
        "validation": audit_dataset(
            validation,
            "validation.csv",
        ),
        "template": audit_dataset(
            template,
            "validation-predictions-template.csv",
        ),
        "december": audit_dataset(
            december,
            "december-chart-inputs.csv",
        ),
    }

    # --------------------------------------------------------
    # Freight-specific checks
    # --------------------------------------------------------

    audit["freight_specific"] = freight_checks(
        train,
        validation,
        template,
        december,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR / "data_quality_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save missing summaries
    # --------------------------------------------------------

    missing_summary(train).to_csv(
        OUTPUT_DIR / "train_missing_values.csv",
        index=False,
    )

    missing_summary(validation).to_csv(
        OUTPUT_DIR / "validation_missing_values.csv",
        index=False,
    )

    missing_summary(template).to_csv(
        OUTPUT_DIR / "template_missing_values.csv",
        index=False,
    )

    missing_summary(december).to_csv(
        OUTPUT_DIR / "december_missing_values.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save numerical summaries
    # --------------------------------------------------------

    numerical_summary(train).to_csv(
        OUTPUT_DIR / "train_numerical_summary.csv",
        index=False,
    )

    numerical_summary(validation).to_csv(
        OUTPUT_DIR / "validation_numerical_summary.csv",
        index=False,
    )

    numerical_summary(december).to_csv(
        OUTPUT_DIR / "december_numerical_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print important findings
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("IMPORTANT FINDINGS")
    print("=" * 70)

    print(
        "\nTrain duplicates:",
        train.duplicated().sum(),
    )

    print(
        "Validation duplicates:",
        validation.duplicated().sum(),
    )

    if "load_id" in train.columns:
        print(
            "Train duplicate load IDs:",
            train["load_id"].duplicated().sum(),
        )

    if "load_id" in validation.columns:
        print(
            "Validation duplicate load IDs:",
            validation["load_id"].duplicated().sum(),
        )

    if "weight" in train.columns:
        print(
            "Train missing weight:",
            train["weight"].isna().sum(),
        )

        print(
            "Train negative weight:",
            (train["weight"] < 0).sum(),
        )

    if "weight" in validation.columns:
        print(
            "Validation missing weight:",
            validation["weight"].isna().sum(),
        )

        print(
            "Validation negative weight:",
            (validation["weight"] < 0).sum(),
        )

    if "market_index" in train.columns:
        print(
            "Train missing market_index:",
            train["market_index"].isna().sum(),
        )

    if "market_index" in validation.columns:
        print(
            "Validation missing market_index:",
            validation["market_index"].isna().sum(),
        )

    if "posted_rate" in train.columns:
        print(
            "Target median:",
            train["posted_rate"].median(),
        )

        print(
            "Target mean:",
            train["posted_rate"].mean(),
        )

    print(
        "\nAudit saved to:",
        OUTPUT_DIR,
    )

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()