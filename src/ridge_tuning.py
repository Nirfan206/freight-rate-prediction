from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def prepare_features(
    train,
    valid
):

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
        feature_columns
    )


def evaluate(
    y_true,
    prediction
):

    prediction = np.maximum(
        prediction,
        1.0
    )

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            prediction
        )
    )

    r2 = r2_score(
        y_true,
        prediction
    )

    return mae, rmse, r2


def main():

    print("=" * 75)
    print("PHASE 8 - RIDGE HYPERPARAMETER TUNING")
    print("=" * 75)

    df = pd.read_csv(
        DATA_DIR / "train-test.csv"
    )

    train, valid = temporal_split(
        df
    )

    print(
        f"\nTraining rows : {len(train):,}"
    )

    print(
        f"Validation rows: {len(valid):,}"
    )

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
        f"\nFeature count: {len(feature_columns)}"
    )

    alphas = [
        0.01,
        0.1,
        1,
        3,
        10,
        30,
        100,
        300,
        1000
    ]

    results = []

    print("\nTesting Ridge alpha values...\n")

    for alpha in alphas:

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
                    alpha=alpha
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

        mae, rmse, r2 = evaluate(
            y_valid,
            prediction
        )

        results.append({
            "alpha": alpha,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(
            f"alpha={alpha:<7} "
            f"MAE={mae:,.3f}  "
            f"RMSE={rmse:,.3f}  "
            f"R2={r2:.5f}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "MAE"
    )

    output_file = (
        OUTPUT_DIR /
        "ridge_tuning_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 75)
    print("RIDGE TUNING RESULTS")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False
        )
    )

    best = results_df.iloc[0]

    print("\n" + "=" * 75)
    print("BEST RIDGE CONFIGURATION")
    print("=" * 75)

    print(
        f"Alpha : {best['alpha']}"
    )

    print(
        f"MAE   : {best['MAE']:,.3f}"
    )

    print(
        f"RMSE  : {best['RMSE']:,.3f}"
    )

    print(
        f"R2    : {best['R2']:.5f}"
    )

    print(
        f"\nSaved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()