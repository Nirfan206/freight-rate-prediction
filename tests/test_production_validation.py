import numpy as np
import pandas as pd

import helpers  # noqa: F401  (sets sys.path)
from helpers import VALID_RECORD
from production.validation import (
    ALLOWED_EQUIPMENT, InputValidationError, normalize_frame, validate_dataframe, validate_record,
)


def errors_for(**changes):
    return validate_record(dict(VALID_RECORD, **changes))


def test_valid_record_passes_with_report_shape():
    report = validate_record(VALID_RECORD).to_dict()
    assert report == {"valid": True, "rows": 1, "errors": [], "warnings": []}


def test_optional_fields_missing_are_warnings_not_errors():
    record = {k: v for k, v in VALID_RECORD.items() if k not in ("weight", "market_index", "quote_signal")}
    report = validate_record(record)
    assert report.valid
    assert any("weight" in w for w in report.warnings)
    assert any("market_index" in w for w in report.warnings)
    assert any("quote_signal" in w for w in report.warnings)


def test_negative_weight_is_warning_not_error():
    report = errors_for(weight=-1200.0)
    assert report.valid
    assert any("weight_negative_rate" in w for w in report.warnings)


def test_non_positive_distance_rejected():
    for bad in (0, -5):
        report = errors_for(distance=bad)
        assert not report.valid and any("distance" in e for e in report.errors)


def test_invalid_coordinates_rejected():
    for column, value in (("pickup_lat", 91), ("delivery_lat", -90.5), ("pickup_lon", 181), ("delivery_lon", -200)):
        report = errors_for(**{column: value})
        assert not report.valid and any(column in e for e in report.errors)


def test_unsupported_equipment_rejected():
    report = errors_for(equipment="Tanker")
    assert not report.valid and any("equipment" in e for e in report.errors)
    for allowed in ALLOWED_EQUIPMENT:
        assert errors_for(equipment=allowed).valid


def test_invalid_date_rejected():
    assert not errors_for(date="not-a-date").valid
    assert not errors_for(date=None).valid


def test_non_numeric_and_non_finite_rejected():
    assert not errors_for(distance="abc").valid
    assert not errors_for(distance=float("inf")).valid
    assert errors_for(weight=float("nan")).valid  # NaN weight is just "missing"
    assert not errors_for(market_index=float("-inf")).valid


def test_duplicate_load_ids_rejected():
    df = pd.DataFrame([VALID_RECORD, dict(VALID_RECORD)])
    report = validate_dataframe(df)
    assert not report.valid and any("duplicate" in e for e in report.errors)


def test_missing_required_column_reported():
    df = pd.DataFrame([{k: v for k, v in VALID_RECORD.items() if k != "equipment"}])
    report = validate_dataframe(df)
    assert not report.valid and any("equipment" in e for e in report.errors)


def test_missing_rate_warning_text_and_high_flag():
    rows = [dict(VALID_RECORD, load_id=f"L{i}", weight=(None if i < 5 else 20000.0)) for i in range(20)]
    report = validate_dataframe(pd.DataFrame(rows))
    assert report.valid
    assert "weight_missing_rate=25.0% (high: >= 20%)" in report.warnings


def test_empty_frame_invalid():
    assert not validate_dataframe(pd.DataFrame()).valid


def test_target_checked_only_when_required():
    df = pd.DataFrame([dict(VALID_RECORD, posted_rate=-5.0)])
    assert validate_dataframe(df).valid
    assert not validate_dataframe(df, require_target=True).valid


def test_normalize_frame_adds_optional_columns_as_nan():
    frame = normalize_frame(pd.DataFrame([{k: v for k, v in VALID_RECORD.items() if k != "weight"}]))
    assert np.isnan(frame.loc[0, "weight"])
    assert frame["distance"].dtype == "float64"


def test_input_validation_error_carries_report():
    report = errors_for(distance=-1)
    err = InputValidationError(report)
    assert err.report is report and "distance" in str(err)


def test_real_development_data_passes_validation():
    from helpers import training_frame
    report = validate_dataframe(training_frame(), require_target=True)
    assert report.valid, report.errors
    assert report.rows == 48000
