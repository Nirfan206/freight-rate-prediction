from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))

from data_cleaning import fit_cleaning_statistics, apply_cleaning
from features import build_features, get_model_features


def metrics(y_true, prediction):

    prediction = np.maximum(prediction, 1.0)

    mae = mean_absolute_error(y_true, prediction)
    rmse = np.sqrt(mean_squared_error(y_true, prediction))
    r2 = r2_score(y_true, prediction)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def temporal_split(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"] < "2025-09-01"].copy()

    valid = df[
        (df["date"] >= "2025-09-01")
        & (df["date"] < "2025-11-01")
    ].copy()

    return train, valid


def equipment_median_baseline(train, valid):

    train = train.copy()
    valid = valid.copy()

    train["rate_per_mile"] = (
        train["posted_rate"] / train["distance"]
    )

    median_rates = (
        train.groupby("equipment")["rate_per_mile"]
        .median()
        .to_dict()
    )

    global_rate = train["rate_per_mile"].median()

    predicted_rpm = (
        valid["equipment"]
        .map(median_rates)
        .fillna(global_rate)
    )

    predictions = predicted_rpm * valid["distance"]

    return predictions


def global_median_baseline(train, valid):

    median_rate = train["posted_rate"].median()

    return np.full(
        len(valid),
        median_rate
    )


def distance_rate_baseline(train, valid):

    train = train.copy()

    train["rate_per_mile"] = (
        train["posted_rate"] / train["distance"]
    )

    median_rpm = train["rate_per_mile"].median()

    return valid["distance"] * median_rpm


def prepare_features(train, valid):

    stats = fit_cleaning_statistics(train)

    train_clean = apply_cleaning(
        train,
        stats
    )

    valid_clean = apply_cleaning(
        valid,
        stats
    )

    train_features = build_features(train_clean)
    valid_features = build_features(valid_clean)

    feature_columns = get_model_features(
        train_features
    )

    X_train = train_features[
        feature_columns
    ]

    X_valid = valid_features[
        feature_columns
    ]

    y_train = train_features["posted_rate"]
    y_valid = valid_features["posted_rate"]

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
        feature_columns
    )


def main():

    print("=" * 75)
    print("PHASE 7 - BASELINES AND MODEL EXPERIMENTS")
    print("=" * 75)

    df = pd.read_csv(
        DATA_DIR / "train-test.csv"
    )

    train, valid = temporal_split(df)

    print(
        f"\nTraining rows : {len(train):,}"
    )

    print(
        f"Validation rows: {len(valid):,}"
    )

    y_valid = valid["posted_rate"]

    results = []

    # ---------------------------------------------------------
    # BASELINE 1
    # ---------------------------------------------------------

    print("\n1. Global median baseline")

    prediction = global_median_baseline(
        train,
        valid
    )

    result = metrics(
        y_valid,
        prediction
    )

    result["Model"] = "Global Median"

    results.append(result)

    print(
        f"MAE = {result['MAE']:,.2f}"
    )

    # ---------------------------------------------------------
    # BASELINE 2
    # ---------------------------------------------------------

    print("\n2. Distance × global median rate/mile")

    prediction = distance_rate_baseline(
        train,
        valid
    )

    result = metrics(
        y_valid,
        prediction
    )

    result["Model"] = "Global Rate Per Mile"

    results.append(result)

    print(
        f"MAE = {result['MAE']:,.2f}"
    )

    # ---------------------------------------------------------
    # BASELINE 3
    # ---------------------------------------------------------

    print("\n3. Equipment median rate/mile")

    prediction = equipment_median_baseline(
        train,
        valid
    )

    result = metrics(
        y_valid,
        prediction
    )

    result["Model"] = "Equipment Rate Per Mile"

    results.append(result)

    print(
        f"MAE = {result['MAE']:,.2f}"
    )

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------

    print("\nPreparing engineered features...")

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        feature_columns
    ) = prepare_features(
        train,
        valid
    )

    print(
        f"Feature count: {len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------

    print("\n4. Random Forest")

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=350,
                max_depth=20,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_valid
    )

    result = metrics(
        y_valid,
        prediction
    )

    result["Model"] = "Random Forest"

    results.append(result)

    print(
        f"MAE = {result['MAE']:,.2f}"
    )

    # ---------------------------------------------------------
    # EXTRA TREES
    # ---------------------------------------------------------

    print("\n5. Extra Trees")

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=350,
                max_depth=24,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_valid
    )

    result = metrics(
        y_valid,
        prediction
    )

    result["Model"] = "Extra Trees"

    results.append(result)

    print(
        f"MAE = {result['MAE']:,.2f}"
    )

    # ---------------------------------------------------------
    # HISTOGRAM GRADIENT BOOSTING
    # ---------------------------------------------------------

    print("\n6. HistGradientBoosting")

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_valid
    )

    result = metrics(
        y_valid,
        prediction
    )

    result["Model"] = "HistGradientBoosting"

    results.append(result)

    print(
        f"MAE = {result['MAE']:,.2f}"
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    ]

    results_df = results_df.sort_values(
        "MAE"
    )

    output_file = (
        OUTPUT_DIR /
        "model_experiment_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 75)
    print("FINAL EXPERIMENT COMPARISON")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n{output_file}"
    )

    print(
        f"\nBest model by MAE: "
        f"{results_df.iloc[0]['Model']}"
    )


if __name__ == "__main__":
    main()