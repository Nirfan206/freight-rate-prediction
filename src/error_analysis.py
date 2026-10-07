from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "experiments"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from data_cleaning import (
    fit_cleaning_statistics,
    apply_cleaning
)

from features import (
    build_features,
    get_model_features
)


def temporal_split(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    train = df[
        df["date"] < "2025-09-01"
    ].copy()

    valid = df[
        (df["date"] >= "2025-09-01")
        & (df["date"] < "2025-11-01")
    ].copy()

    return train, valid


def prepare_data(train, valid):

    stats = fit_cleaning_statistics(
        train
    )

    train_clean = apply_cleaning(
        train,
        stats
    )

    valid_clean = apply_cleaning(
        valid,
        stats
    )

    train_features = build_features(
        train_clean
    )

    valid_features = build_features(
        valid_clean
    )

    feature_columns = get_model_features(
        train_features
    )

    X_train = train_features[
        feature_columns
    ].copy()

    X_valid = valid_features[
        feature_columns
    ].copy()

    y_train = train_features[
        "posted_rate"
    ].copy()

    y_valid = valid_features[
        "posted_rate"
    ].copy()

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
        valid_features
    )


def calculate_group_metrics(
    df,
    group_column
):

    result = (
        df.groupby(group_column)
        .agg(
            rows=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            RMSE=(
                "squared_error",
                lambda x: np.sqrt(x.mean())
            ),
            actual_mean=("actual_rate", "mean"),
            predicted_mean=("predicted_rate", "mean")
        )
        .reset_index()
    )

    return result.sort_values(
        "MAE",
        ascending=False
    )


def main():

    print("=" * 75)
    print("PHASE 9 - ERROR ANALYSIS")
    print("=" * 75)

    df = pd.read_csv(
        DATA_DIR / "train-test.csv"
    )

    train, valid = temporal_split(
        df
    )

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        valid_features
    ) = prepare_data(
        train,
        valid
    )

    # ---------------------------------------------------------
    # TRAIN FINAL TEMPORAL RIDGE
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

    model.fit(
        X_train,
        np.log1p(y_train)
    )

    log_prediction = model.predict(
        X_valid
    )

    prediction = np.expm1(
        log_prediction
    )

    prediction = np.maximum(
        prediction,
        1.0
    )

    # ---------------------------------------------------------
    # ERROR DATAFRAME
    # ---------------------------------------------------------

    errors = valid.copy()

    errors["actual_rate"] = y_valid.values

    errors["predicted_rate"] = prediction

    errors["absolute_error"] = (
        errors["actual_rate"]
        - errors["predicted_rate"]
    ).abs()

    errors["percentage_error"] = (
        errors["absolute_error"]
        / errors["actual_rate"].clip(lower=1)
        * 100
    )

    errors["squared_error"] = (
        errors["actual_rate"]
        - errors["predicted_rate"]
    ) ** 2

    errors["rate_per_mile_actual"] = (
        errors["actual_rate"]
        / errors["distance"]
    )

    errors["rate_per_mile_predicted"] = (
        errors["predicted_rate"]
        / errors["distance"]
    )

    # ---------------------------------------------------------
    # OVERALL
    # ---------------------------------------------------------

    overall = pd.DataFrame([
        {
            "rows": len(errors),
            "MAE": errors["absolute_error"].mean(),
            "RMSE": np.sqrt(
                errors["squared_error"].mean()
            ),
            "MAPE_percent": errors[
                "percentage_error"
            ].mean()
        }
    ])

    overall.to_csv(
        OUTPUT_DIR / "error_overall.csv",
        index=False
    )

    # ---------------------------------------------------------
    # EQUIPMENT
    # ---------------------------------------------------------

    equipment = calculate_group_metrics(
        errors,
        "equipment"
    )

    equipment.to_csv(
        OUTPUT_DIR / "error_by_equipment.csv",
        index=False
    )

    # ---------------------------------------------------------
    # DISTANCE BUCKET
    # ---------------------------------------------------------

    errors["distance_bucket"] = pd.cut(
        errors["distance"],
        bins=[
            0,
            200,
            500,
            1000,
            1500,
            2000,
            np.inf
        ],
        labels=[
            "<=200",
            "201-500",
            "501-1000",
            "1001-1500",
            "1501-2000",
            ">2000"
        ]
    )

    distance_result = calculate_group_metrics(
        errors,
        "distance_bucket"
    )

    distance_result.to_csv(
        OUTPUT_DIR / "error_by_distance.csv",
        index=False
    )

    # ---------------------------------------------------------
    # WEIGHT BUCKET
    # ---------------------------------------------------------

    errors["weight_bucket"] = pd.cut(
        errors["weight"],
        bins=[
            -np.inf,
            10000,
            20000,
            30000,
            40000,
            np.inf
        ],
        labels=[
            "<=10k",
            "10k-20k",
            "20k-30k",
            "30k-40k",
            ">40k"
        ]
    )

    weight_result = calculate_group_metrics(
        errors,
        "weight_bucket"
    )

    weight_result.to_csv(
        OUTPUT_DIR / "error_by_weight.csv",
        index=False
    )

    # ---------------------------------------------------------
    # MONTH / DATE
    # ---------------------------------------------------------

    errors["month"] = (
        pd.to_datetime(errors["date"])
        .dt.month
    )

    monthly_result = calculate_group_metrics(
        errors,
        "month"
    )

    monthly_result.to_csv(
        OUTPUT_DIR / "error_by_month.csv",
        index=False
    )

    # ---------------------------------------------------------
    # NEW CITY ANALYSIS
    # ---------------------------------------------------------

    train_pickups = set(
        train["pickup"].unique()
    )

    train_deliveries = set(
        train["delivery"].unique()
    )

    errors["new_pickup"] = (
        ~errors["pickup"].isin(
            train_pickups
        )
    )

    errors["new_delivery"] = (
        ~errors["delivery"].isin(
            train_deliveries
        )
    )

    errors["new_city"] = (
        errors["new_pickup"]
        | errors["new_delivery"]
    )

    new_city_result = calculate_group_metrics(
        errors,
        "new_city"
    )

    new_city_result.to_csv(
        OUTPUT_DIR / "error_new_city.csv",
        index=False
    )

    # ---------------------------------------------------------
    # TOP ERRORS
    # ---------------------------------------------------------

    top_errors = errors.sort_values(
        "absolute_error",
        ascending=False
    ).head(100)

    top_errors.to_csv(
        OUTPUT_DIR / "top_100_errors.csv",
        index=False
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\nOverall:")
    print(
        overall.to_string(
            index=False
        )
    )

    print("\nBy equipment:")
    print(
        equipment.to_string(
            index=False
        )
    )

    print("\nBy distance:")
    print(
        distance_result.to_string(
            index=False
        )
    )

    print("\nBy weight:")
    print(
        weight_result.to_string(
            index=False
        )
    )

    print("\nBy new-city status:")
    print(
        new_city_result.to_string(
            index=False
        )
    )

    print(
        "\nTop 10 largest errors:"
    )

    print(
        top_errors[
            [
                "load_id",
                "pickup",
                "delivery",
                "distance",
                "equipment",
                "actual_rate",
                "predicted_rate",
                "absolute_error",
                "percentage_error"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print("\nSaved error-analysis files to:")

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()