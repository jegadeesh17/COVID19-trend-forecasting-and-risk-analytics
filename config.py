import os

APP_TITLE = "COVID-19 Predictive Analytics"
APP_SUBTITLE = "Global Outbreak Intelligence & Forecasting Engine"

BASE_DIR = os.path.dirname(__file__)

# File paths mapped to project workspace dynamically
DATA_PATH = os.path.join(BASE_DIR, "data", "compact.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model Input Features
FEATURES = [
    'total_cases', 'cases_7d_avg', 'deaths_7d_avg', 'growth_rate', 
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
