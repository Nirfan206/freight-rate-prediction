"""
Exploratory Data Analysis for the Spotter Freight Rate Prediction project.

Generates business-relevant charts and summary statistics from the
training dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "train-test.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "validation.csv"

EDA_DIR = PROJECT_ROOT / "outputs" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train["date"] = pd.to_datetime(train["date"])
validation["date"] = pd.to_datetime(validation["date"])


# ============================================================
# Derived variables
# ============================================================

train["rate_per_mile"] = (
    train["posted_rate"] / train["distance"]
)

train["month"] = train["date"].dt.to_period("M").astype(str)

train["day_of_week"] = train["date"].dt.day_name()

train["distance_bucket"] = pd.cut(
    train["distance"],
    bins=[
        0,
        200,
        500,
        1000,
        1500,
        2000,
        np.inf,
    ],
    labels=[
        "<=200",
        "201-500",
        "501-1000",
        "1001-1500",
        "1501-2000",
        "2000+",
    ],
)


# ============================================================
# Helper
# ============================================================

def save_plot(filename: str) -> None:
    """Save and close the current matplotlib figure."""

    plt.tight_layout()
    plt.savefig(
        EDA_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 1. Posted rate distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    train["posted_rate"],
    bins=80,
)

plt.xlabel("Posted Rate")
plt.ylabel("Number of Loads")
plt.title("Distribution of Posted Freight Rates")

save_plot("01_posted_rate_distribution.png")


# ============================================================
# 2. Log target distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    np.log1p(train["posted_rate"]),
    bins=80,
)

plt.xlabel("log1p(Posted Rate)")
plt.ylabel("Number of Loads")
plt.title("Distribution of log1p(Posted Rate)")

save_plot("02_log_posted_rate_distribution.png")


# ============================================================
# 3. Rate per mile distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    train["rate_per_mile"],
    bins=100,
)

plt.xlim(
    0,
    train["rate_per_mile"].quantile(0.99),
)

plt.xlabel("Rate per Mile")
plt.ylabel("Number of Loads")
plt.title("Distribution of Freight Rate per Mile")

save_plot("03_rate_per_mile_distribution.png")


# ============================================================
# 4. Posted rate vs distance
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    train["distance"],
    train["posted_rate"],
    s=8,
    alpha=0.25,
)

plt.xlabel("Distance (miles)")
plt.ylabel("Posted Rate")
plt.title("Posted Rate vs Distance")

save_plot("04_posted_rate_vs_distance.png")


# ============================================================
# 5. Rate per mile vs distance
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    train["distance"],
    train["rate_per_mile"],
    s=8,
    alpha=0.25,
)

plt.ylim(
    0,
    train["rate_per_mile"].quantile(0.99),
)

plt.xlabel("Distance (miles)")
plt.ylabel("Rate per Mile")
plt.title("Rate per Mile vs Distance")

save_plot("05_rate_per_mile_vs_distance.png")


# ============================================================
# 6. Equipment comparison
# ============================================================

equipment_rate = (
    train.groupby("equipment")["posted_rate"]
    .median()
    .sort_values()
)

plt.figure(figsize=(9, 6))

equipment_rate.plot(
    kind="bar",
)

plt.xlabel("Equipment")
plt.ylabel("Median Posted Rate")
plt.title("Median Posted Rate by Equipment")

save_plot("06_median_rate_by_equipment.png")


# ============================================================
# 7. Rate per mile by equipment
# ============================================================

equipment_rpm = (
    train.groupby("equipment")["rate_per_mile"]
    .median()
    .sort_values()
)

plt.figure(figsize=(9, 6))

equipment_rpm.plot(
    kind="bar",
)

plt.xlabel("Equipment")
plt.ylabel("Median Rate per Mile")
plt.title("Median Rate per Mile by Equipment")

save_plot("07_rate_per_mile_by_equipment.png")


# ============================================================
# 8. Weight vs posted rate
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    train["weight"],
    train["posted_rate"],
    s=8,
    alpha=0.25,
)

plt.xlabel("Weight")
plt.ylabel("Posted Rate")
plt.title("Posted Rate vs Weight")

save_plot("08_posted_rate_vs_weight.png")


# ============================================================
# 9. Market index over time
# ============================================================

market_daily = (
    train.groupby("date")["market_index"]
    .mean()
)

plt.figure(figsize=(11, 6))

plt.plot(
    market_daily.index,
    market_daily.values,
)

plt.xlabel("Date")
plt.ylabel("Market Index")
plt.title("Market Index Over Time")

plt.xticks(rotation=45)

save_plot("09_market_index_over_time.png")


# ============================================================
# 10. Quote signal distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    train["quote_signal"],
    bins=60,
)

plt.xlabel("Quote Signal")
plt.ylabel("Number of Loads")
plt.title("Distribution of Quote Signal")

save_plot("10_quote_signal_distribution.png")


# ============================================================
# 11. Median posted rate by month
# ============================================================

monthly_rate = (
    train.groupby("month")["posted_rate"]
    .median()
)

plt.figure(figsize=(11, 6))

monthly_rate.plot(
    marker="o",
)

plt.xlabel("Month")
plt.ylabel("Median Posted Rate")
plt.title("Median Posted Rate by Month")

plt.xticks(rotation=45)

save_plot("11_monthly_median_rate.png")


# ============================================================
# 12. Median rate per mile by distance bucket
# ============================================================

distance_rpm = (
    train.groupby(
        "distance_bucket",
        observed=True,
    )["rate_per_mile"]
    .median()
)

plt.figure(figsize=(10, 6))

distance_rpm.plot(
    kind="bar",
)

plt.xlabel("Distance Bucket")
plt.ylabel("Median Rate per Mile")
plt.title("Median Rate per Mile by Distance Bucket")

save_plot("12_rate_per_mile_by_distance_bucket.png")


# ============================================================
# 13. Median rate by day of week
# ============================================================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

dow_rate = (
    train.groupby("day_of_week")["posted_rate"]
    .median()
    .reindex(day_order)
)

plt.figure(figsize=(10, 6))

dow_rate.plot(
    kind="bar",
)

plt.xlabel("Day of Week")
plt.ylabel("Median Posted Rate")
plt.title("Median Posted Rate by Day of Week")

save_plot("13_rate_by_day_of_week.png")


# ============================================================
# 14. Market index vs posted rate
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    train["market_index"],
    train["posted_rate"],
    s=8,
    alpha=0.25,
)

plt.xlabel("Market Index")
plt.ylabel("Posted Rate")
plt.title("Posted Rate vs Market Index")

save_plot("14_posted_rate_vs_market_index.png")


# ============================================================
# 15. Quote signal vs posted rate
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    train["quote_signal"],
    train["posted_rate"],
    s=8,
    alpha=0.25,
)

plt.xlabel("Quote Signal")
plt.ylabel("Posted Rate")
plt.title("Posted Rate vs Quote Signal")

save_plot("15_posted_rate_vs_quote_signal.png")


# ============================================================
# 16. Train vs validation distance distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    train["distance"],
    bins=60,
    alpha=0.5,
    label="Train",
)

plt.hist(
    validation["distance"],
    bins=60,
    alpha=0.5,
    label="Validation",
)

plt.xlabel("Distance")
plt.ylabel("Number of Loads")
plt.title("Train vs Validation Distance Distribution")
plt.legend()

save_plot("16_train_vs_validation_distance.png")


# ============================================================
# 17. Train vs validation weight distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    train["weight"].dropna(),
    bins=60,
    alpha=0.5,
    label="Train",
)

plt.hist(
    validation["weight"].dropna(),
    bins=60,
    alpha=0.5,
    label="Validation",
)

plt.xlabel("Weight")
plt.ylabel("Number of Loads")
plt.title("Train vs Validation Weight Distribution")
plt.legend()

save_plot("17_train_vs_validation_weight.png")


# ============================================================
# 18. Correlation analysis
# ============================================================

numeric_columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate",
    "rate_per_mile",
]

correlation = train[numeric_columns].corr(
    method="spearman"
)

correlation.to_csv(
    EDA_DIR / "spearman_correlation.csv"
)


# ============================================================
# Summary statistics
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "train_rows",
        "validation_rows",
        "target_mean",
        "target_median",
        "target_std",
        "target_min",
        "target_max",
        "rate_per_mile_mean",
        "rate_per_mile_median",
        "rate_per_mile_min",
        "rate_per_mile_max",
        "distance_mean",
        "distance_median",
        "weight_mean",
        "weight_median",
        "market_index_mean",
        "quote_signal_mean",
    ],
    "value": [
        len(train),
        len(validation),
        train["posted_rate"].mean(),
        train["posted_rate"].median(),
        train["posted_rate"].std(),
        train["posted_rate"].min(),
        train["posted_rate"].max(),
        train["rate_per_mile"].mean(),
        train["rate_per_mile"].median(),
        train["rate_per_mile"].min(),
        train["rate_per_mile"].max(),
        train["distance"].mean(),
        train["distance"].median(),
        train["weight"].mean(),
        train["weight"].median(),
        train["market_index"].mean(),
        train["quote_signal"].mean(),
    ],
})

summary.to_csv(
    EDA_DIR / "eda_summary.csv",
    index=False,
)


# ============================================================
# Console output
# ============================================================

print("=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print(f"\nTraining rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")

print(
    f"\nPosted rate median: "
    f"{train['posted_rate'].median():,.2f}"
)

print(
    f"Posted rate mean: "
    f"{train['posted_rate'].mean():,.2f}"
)

print(
    f"\nRate/mile median: "
    f"{train['rate_per_mile'].median():.4f}"
)

print(
    f"Distance median: "
    f"{train['distance'].median():,.2f}"
)

print(
    f"Weight median: "
    f"{train['weight'].median():,.2f}"
)

print("\nSpearman correlation with posted_rate:")

print(
    train[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    ]
    .corr(method="spearman")["posted_rate"]
    .sort_values(ascending=False)
)

print(
    "\nEDA files saved to:"
    f"\n{EDA_DIR}"
)