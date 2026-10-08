# Production architecture

Scope: a deployable architecture and local production simulation layered on the offline assessment pipeline.
No public deployment, authentication, autoscaling or real-time alerting is provided.

## Request path

```
POST /predict (JSON, raw features)
  -> Pydantic schema (types, ranges, equipment, finite numbers, date)
  -> production.validation (same rules, batch-capable)      -> 422 on fundamentally invalid input
  -> data_cleaning.apply_cleaning   (assessment code, unchanged; training-fitted statistics)
  -> features.build_features        (assessment code, unchanged)
  -> Pipeline(imputer, scaler, Ridge) on log1p target -> expm1 -> floor at 1.0
  -> empirical interval (lower_rate / upper_rate)
  -> JSON response + optional JSONL monitoring record
```

Validation policy: **reject** non-positive distance, impossible coordinates, unknown equipment, invalid dates,
non-numeric/non-finite values, duplicate load IDs. **Warn, don't reject**: missing/negative weight, missing market
index, missing quote signal (the assessment cleaning layer handles them).

## Configuration (`src/production/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_REGISTRY_PATH` | `models/registry` | registry root |
| `PRODUCTION_MODEL_VERSION` | (empty) | force a version, overriding `production.json` |
| `MONITORING_PATH` | `monitoring` | logs and reports |
| `DATA_DIR` | `data` | training data location |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |
| `API_HOST`, `API_PORT` | `0.0.0.0`, `8000` | server bind |
| `PREDICTION_LOGGING` | `true` | write `predictions.jsonl` |
| `LOG_INPUT_FEATURES` | `true` | include freight features in records (required for live drift) |

## Registry

See `models/registry/README.md`. Promotion/rollback checks: model file exists, metadata complete, metrics present,
`feature_count` equals the current feature pipeline's output, and a smoke prediction succeeds.

## Prediction intervals

Residuals `log1p(actual) - log1p(predicted)` from the temporal holdout (trained Jan-Aug, scored Sep-Oct 2025); the 5th/95th
percentiles are applied multiplicatively. **These are empirical residual-based prediction intervals, not calibrated
probabilistic confidence intervals.** The stored `holdout_check` calibrates on the first half of the holdout and
measures coverage on the second half (90% nominal; 87.5% observed for v1), so expect mild under-coverage under shift.
The official assessment predictions are unaffected and carry no interval.

## Monitoring

* **Predictions**: `monitoring/predictions/predictions.jsonl`, one JSON object per line. `event` is `prediction`
  (timestamp, load_id, predicted_rate, lower/upper, model_version, latency_ms, selected inputs) or `error`
  (type and status code only, never payloads). No secrets or free text are logged; log writes never fail a request.
* **Drift** (`src/production/drift.py`): PSI over reference-quantile bins (numeric) and over category frequencies
  (equipment, unseen categories pooled as `__other__`). Thresholds: **PSI < 0.10 stable; 0.10-0.25 warning; > 0.25 drift**;
  fewer than 30 values = `insufficient_data`. Drift means inputs moved, not that the model is wrong.
* **Performance** (`performance.py`): MAE, RMSE, MAPE, bias (predicted - actual) overall and by equipment, distance bucket
  and month; baseline = holdout metrics in the model metadata; MAE ratio > 1.10 warning, > 1.25 degraded; fewer than 30
  matched rows = `insufficient_data`. With no actuals: `actual target data unavailable`.
* **Retraining policy** (`retraining.py`): recommend if >= 2 features drift, performance is degraded, or the model is older
  than 90 days. Execution is always explicit; new versions are never auto-promoted.

## Logging

JSON lines on stderr (`timestamp, level, logger, message, event, ...`). Keys containing `api_key`, `password`, `secret`,
`token`, `authorization`, `credential` are redacted. Tracebacks go to server logs only; clients get a generic 500.

## CI/CD

`tests.yml` (push, PR): install, import checks, `pytest -q`, official scorer. `production-build.yml`: tests, Docker build,
container `/health`, `/model/info`, `/predict` smoke check. No deployment, no cloud credentials.

## Known limitations

* API loads the model at startup; promote/rollback needs a restart (no unauthenticated admin endpoint by design).
* Pickled models are scikit-learn-version specific; registry files must come from a trusted source.
* Monitoring is file based and batch (run the scripts); there is no scheduler or alerting.
* No production traffic or actual rates exist yet; no production accuracy is claimed.
* `requirements.txt` pins the original scientific stack exactly but gives version *ranges* for the four new API packages.
