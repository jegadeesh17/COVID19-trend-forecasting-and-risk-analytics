# COVID Predictive Analytics — Evaluation Report

## Methodology
- Time-series features: lags, 7-day averages, growth and mortality rates
- Models compared: Linear Regression, Random Forest, XGBoost
- Holdout evaluation on engineered `new_cases` target
- Production model: tuned XGBoost (`models/covid_xgb_model_tuned.joblib`)

## Metrics (holdout)

| Model | MAE | RMSE | R² | CV R² (mean) |
|-------|-----|------|-----|--------------|
| Linear Regression | 7.93 | 95.49 | 0.955 | 0.995 |
| Random Forest | 6.40 | 88.84 | 0.961 | 0.807 |
| XGBoost | 6.40 | 88.04 | 0.962 | 0.803 |

**Selected for deployment:** XGBoost (best holdout R²).

## Limitations
- Metrics reflect pipeline validity on historical COVID aggregates; not a clinical forecasting system.
- Sample data in git may produce different numbers than full-dataset retraining.
- Source metrics: `models/metrics.json` — regenerate after retraining.
