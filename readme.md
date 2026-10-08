# Spotter Freight Rate Prediction

Machine learning solution for predicting freight posted rates from shipment, geographic, equipment, distance, weight, market, and quote-related features.

## Project Overview

This project develops a supervised machine learning regression pipeline for freight-rate prediction.

The workflow includes:

1. Data quality auditing
2. Exploratory data analysis
3. Data cleaning
4. Feature engineering
5. Temporal validation
6. Baseline comparison
7. Model experimentation
8. Ridge hyperparameter tuning
9. Error analysis
10. Final model training
11. Validation prediction generation
12. December scenario prediction
13. Automated validation tests

---

## Dataset

The development dataset contains:

- 48,000 labeled shipment records
- 14 original columns
- January 1, 2025 through October 31, 2025

The assessment validation dataset contains:

- 12,000 shipment records
- November 1, 2025 through December 31, 2025

The target variable is:

```text
posted_rate
```

---

## Production ML layer (additive)

> The assessment solution is an **offline batch ML pipeline** and is unchanged. The components below are an
> *additional* deployable architecture and **local production simulation**. No public production deployment is claimed.

| Area | Where |
|---|---|
| Prediction API (FastAPI) | `api/` |
| Production code (config, registry, predictor, monitoring, drift, performance, uncertainty, retraining) | `src/production/` |
| Model registry (v1 = the assessment artifact) | `models/registry/` |
| Operational scripts | `scripts/` |
| Docker / compose | `docker/Dockerfile`, `docker-compose.yml` |
| CI | `.github/workflows/` |
| Full documentation | [`docs/PRODUCTION.md`](docs/PRODUCTION.md) |

### Run the API

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> (Swagger) or `/redoc`.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "load_id": "LOAD_001", "date": "2025-11-15", "distance": 850, "weight": 32000,
  "pickup_lat": 40.7128, "pickup_lon": -74.0060, "delivery_lat": 41.8781, "delivery_lon": -87.6298,
  "equipment": "Dry Van", "market_index": 1.05, "quote_signal": 2.0}'
```

`weight`, `market_index` and `quote_signal` may be omitted. The response contains `predicted_rate`, `lower_rate`,
`upper_rate` (an **empirical residual-based prediction interval, not a calibrated confidence interval**), and `model_version`.

### Docker

```bash
docker build -f docker/Dockerfile -t freight-rate-api .
docker run --rm -p 8000:8000 freight-rate-api          # http://localhost:8000/health
# or
docker compose up --build
```

### Model lifecycle

```bash
python scripts/tune_model.py                      # alpha search -> outputs/experiments/production_tuning.csv
python scripts/train_production.py                # registers the next version (does NOT promote)
python scripts/promote_model.py  --version v2     # validated promotion
python scripts/rollback_model.py --version v1     # roll back (nothing is deleted)
```

Restart the API after promote/rollback (the model is loaded at startup).

### Monitoring

```bash
python scripts/monitor_predictions.py                         # summary of monitoring/predictions/predictions.jsonl
python scripts/detect_drift.py                                # drift of logged inputs vs training data (PSI)
python scripts/evaluate_production_model.py --actuals a.csv   # needs load_id,actual_rate
python scripts/train_production.py --check-only               # is retraining recommended?
python dashboard/build_dashboard.py                           # includes the "Production ML" page
```

### Configuration

Environment variables (see `.env.example`): `MODEL_REGISTRY_PATH`, `PRODUCTION_MODEL_VERSION`, `MONITORING_PATH`,
`LOG_LEVEL`, `API_HOST`, `API_PORT`, `PREDICTION_LOGGING`, `LOG_INPUT_FEATURES`. No secrets are required.

### Assessment compatibility

```bash
pytest -q
python score.py --predictions validation_predictions.csv --december-predictions outputs/predictions/december_predictions.csv
```

The production layer never modifies `validation_predictions.csv`, `outputs/predictions/`, `artifacts/`, or `data/`.
