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