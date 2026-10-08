import numpy as np
import pandas as pd

import helpers  # noqa: F401
from helpers import training_frame
from production.drift import (
    NUMERIC_FEATURES, build_reference_profile, detect_drift, population_stability_index, psi_status,
)


def synthetic(n=5000, seed=0, shift=0.0, equipment=("Dry Van", "Reefer", "Flatbed"), p=(0.6, 0.25, 0.15)):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(100 + shift, 15, n) for c in NUMERIC_FEATURES})
    df["equipment"] = rng.choice(equipment, n, p=p)
    return df


def by_feature(report):
    return {f["feature"]: f for f in report["features"]}


def test_psi_zero_for_identical_distributions():
    assert population_stability_index([0.25] * 4, [0.25] * 4) == 0.0


def test_psi_known_value():
    ref, cur = [0.5, 0.5], [0.7, 0.3]
    expected = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    assert abs(population_stability_index(ref, cur) - expected) < 1e-12


def test_threshold_boundaries():
    assert psi_status(0.0) == "stable"
    assert psi_status(0.0999) == "stable"
    assert psi_status(0.10) == "warning"
    assert psi_status(0.25) == "warning"
    assert psi_status(0.2501) == "drift"


def test_reference_profile_covers_all_monitored_features():
    profile = build_reference_profile(synthetic())
    assert set(NUMERIC_FEATURES) | {"equipment"} == set(profile["features"])
    for name in NUMERIC_FEATURES:
        assert abs(sum(profile["features"][name]["proportions"]) - 1.0) < 1e-9


def test_same_distribution_is_stable():
    profile = build_reference_profile(synthetic(seed=0))
    report = detect_drift(synthetic(seed=1), profile)
    assert report["overall_status"] == "stable"
    assert all(f["status"] == "stable" for f in report["features"])


def test_shifted_distribution_is_flagged_as_drift():
    profile = build_reference_profile(synthetic(seed=0))
    report = detect_drift(synthetic(seed=1, shift=30.0), profile)
    f = by_feature(report)
    assert f["distance"]["status"] == "drift" and f["distance"]["metric"] == "PSI"
    assert report["overall_status"] == "drift"


def test_result_shape_matches_spec():
    profile = build_reference_profile(synthetic())
    f = by_feature(detect_drift(synthetic(seed=2), profile))["distance"]
    assert {"feature", "metric", "value", "status"} <= set(f)


def test_categorical_frequency_shift_and_unseen_category():
    profile = build_reference_profile(synthetic())
    shifted = detect_drift(synthetic(seed=3, p=(0.2, 0.2, 0.6)), profile)
    eq = by_feature(shifted)["equipment"]
    assert eq["status"] == "drift" and eq["max_abs_frequency_diff"] > 0.3
    assert set(eq["frequencies"]) == {"Dry Van", "Reefer", "Flatbed"}
    unseen = detect_drift(synthetic(seed=4, equipment=("Tanker",), p=(1.0,)), profile)
    assert by_feature(unseen)["equipment"]["status"] == "drift"
    assert "__other__" in by_feature(unseen)["equipment"]["frequencies"]


def test_insufficient_data_is_not_called_stable():
    profile = build_reference_profile(synthetic())
    report = detect_drift(synthetic(n=10), profile)
    assert all(f["status"] == "insufficient_data" and f["value"] is None for f in report["features"])
    assert report["overall_status"] == "insufficient_data"


def test_missing_values_are_ignored_for_numeric_psi():
    profile = build_reference_profile(synthetic(seed=0))
    cur = synthetic(seed=1)
    cur.loc[:500, "weight"] = np.nan
    f = by_feature(detect_drift(cur, profile))["weight"]
    assert f["status"] == "stable" and f["current_missing_rate"] > 0.09


def test_real_training_data_resample_is_stable():
    df = training_frame()
    profile = build_reference_profile(df)
    report = detect_drift(df.sample(4000, random_state=7), profile)
    assert report["overall_status"] == "stable"
