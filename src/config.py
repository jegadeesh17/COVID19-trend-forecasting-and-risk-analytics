import os

APP_TITLE = "COVID-19 Predictive Analytics"
APP_SUBTITLE = "Global Outbreak Intelligence & Forecasting Engine"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# File paths mapped to project workspace dynamically
_data_dir = os.path.join(BASE_DIR, "data")

def _first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[-1]

DATA_PATH = _first_existing(
    os.path.join(_data_dir, "compact.csv"),
    os.path.join(_data_dir, "compact.csv.gz"),
    os.path.join(_data_dir, "compact_sample.csv"),
)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model Input Features
FEATURES = [
    'total_cases_lag1', 'cases_7d_avg', 'deaths_7d_avg', 'growth_rate', 
    'mortality_rate', 'month', 'week', 'lag_1', 'lag_7', 
    'population_density', 'gdp_per_capita', 'diabetes_prevalence'
]

TARGET = 'new_cases'

# Premium Color Palette Tokens
COLORS = {
    "primary": "#3B82F6",
    "danger": "#EF4444",
    "success": "#10B981",
    "warning": "#F59E0B",
    "background": "rgba(255, 255, 255, 0.05)"
}
