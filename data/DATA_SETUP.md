# Data Setup

## Included in Git (demo / dashboard)

| File | Purpose |
|------|---------|
| `compact_sample.csv` | Subset for EDA, feature engineering, and Streamlit |
| `cleaned_covid_data_sample.parquet` | Pre-cleaned sample parquet |

## Full dataset (local only)

| File | Purpose |
|------|---------|
| `compact.csv` or `compact.csv.gz` | Full global COVID time-series |
| `cleaned_covid_data.parquet` | Full cleaned parquet after notebook ETL |

**How to obtain:** Run the notebook ETL on your full source export, or restore from your original `compact.csv` archive.

**Resolution order:** `src/config.py` tries full files first (`compact.csv`, `compact.csv.gz`), then `compact_sample.csv`.

**Retrain:** Re-run the training cells in `notebooks/Global COVID Predictive Analytics System.ipynb` or your `src/` training script.
