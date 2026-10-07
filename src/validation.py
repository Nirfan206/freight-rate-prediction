from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent))

from data_cleaning import fit_cleaning_statistics, apply_cleaning
from features import build_features, get_model_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_predictions(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1.0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def temporal_split(df):
    """
    Temporal holdout:
    Training: January-August 2025
    Validation: September-October 2025
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_mask = df["date"] < "2025-09-01"
    valid_mask = (
        (df["date"] >= "2025-09-01")
        & (df["date"] < "2025-11-01")
    )

    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    return train_df, valid_df


def prepare_data(train_df, valid_df):
    """
    Fit cleaning statistics ONLY on temporal training data.
    """

    stats = fit_cleaning_statistics(train_df)

    train_clean = apply_cleaning(train_df, stats)
    valid_clean = apply_cleaning(valid_df, stats)

    train_features = build_features(train_clean)
    valid_features = build_features(valid_clean)

    feature_columns = get_model_features(train_features)

    X_train = train_features[feature_columns].copy()
    X_valid = valid_features[feature_columns].copy()

    y_train = train_features["posted_rate"].copy()
    y_valid = valid_features["posted_rate"].copy()

    return X_train, X_valid, y_train, y_valid, feature_columns


def build_models():
    models = {

        "Ridge_LogTarget": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),

        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=18,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=250,
                    max_depth=20,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ])
    }

    return models


def main():

    print("=" * 70)
    print("PHASE 6 - TEMPORAL VALIDATION")
    print("=" * 70)

    train_path = DATA_DIR / "train-test.csv"

    df = pd.read_csv(train_path)

    print(f"\nFull development data: {df.shape}")

    train_df, valid_df = temporal_split(df)

    print(f"Temporal training rows : {len(train_df):,}")
    print(f"Temporal validation rows: {len(valid_df):,}")

    print(
        f"\nTraining period: "
        f"{train_df['date'].min()} -> {train_df['date'].max()}"
    )

    print(
        f"Validation period: "
        f"{valid_df['date'].min()} -> {valid_df['date'].max()}"
    )

    X_train, X_valid, y_train, y_valid, features = prepare_data(
        train_df,
        valid_df
    )

    print(f"\nNumber of model features: {len(features)}")

    print("\nTraining models...")

    models = build_models()

    results = []

    for name, model in models.items():

        print(f"\nRunning: {name}")

        if name == "Ridge_LogTarget":

            model.fit(
                X_train,
                np.log1p(y_train)
            )

            log_pred = model.predict(X_valid)

            predictions = np.expm1(log_pred)

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(X_valid)

        metrics = evaluate_predictions(
            y_valid,
            predictions
        )

        metrics["Model"] = name

        results.append(metrics)

        print(f"MAE : {metrics['MAE']:,.2f}")
        print(f"RMSE: {metrics['RMSE']:,.2f}")
        print(f"R2  : {metrics['R2']:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df[
        ["Model", "MAE", "RMSE", "R2"]
    ]

    results_df = results_df.sort_values(
        "MAE",
        ascending=True
    )

    output_file = OUTPUT_DIR / "temporal_validation_results.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("TEMPORAL VALIDATION RESULTS")
    print("=" * 70)

    print(results_df.to_string(index=False))

    print(
        f"\nResults saved to:\n{output_file}"
    )

    print("\nBest model by MAE:")
    print(results_df.iloc[0]["Model"])


if __name__ == "__main__":
    main()