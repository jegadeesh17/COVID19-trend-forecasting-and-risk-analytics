import os
import json
import joblib
import streamlit as st
import pandas as pd
from src.config import DATA_PATH, FEATURES, TARGET, MODEL_DIR, BASE_DIR
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.modeling import train_models

@st.cache_data(show_spinner="Accessing clean health records cache...")
def load_and_prep_data():
    """Load data bundle, interpolate missing points, and construct dynamic lag features."""
    parquet_path = os.path.join(BASE_DIR, "data", "cleaned_covid_data.parquet")

    # Fast path: load compiled columnar data if available
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file missing at requested path: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_data(raw_df)
    engineered_df = create_features(processed_df)
    return engineered_df

@st.cache_resource(show_spinner="Loading models...")
def load_cached_models():
    """Load pre-trained model artifacts or train from scratch as fallback."""
    xgb_path = os.path.join(MODEL_DIR, "covid_xgb_model_tuned.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    # Check if all pre-trained artifacts exist
    if os.path.exists(xgb_path) and os.path.exists(scaler_path) and os.path.exists(metrics_path):
        try:
            xgb_model = joblib.load(xgb_path)
            scaler = joblib.load(scaler_path)

            # Load real evaluation metrics saved during training
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            trained_models = {"XGBoost": xgb_model}
            results_df = pd.DataFrame(metrics)

            return trained_models, results_df, None, None, scaler
        except Exception:
            # Fall through to retraining if artifacts are corrupt or incompatible
            pass

    # Fallback: train models from data
    df = load_and_prep_data()
    # Sort globally by date for proper chronological splitting
    df = df.sort_values('date')

    # Sample recent records if dataset is very large, preserving chronological order
    if len(df) > 10000:
        df = df.tail(10000)

    X = df[FEATURES]
    y = df[TARGET]
    dates = df['date']

    trained_models, results_df, X_test_scaled, y_test, scaler = train_models(
        X, y, dates=dates
    )

    # Persist artifacts for future fast loading
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_name = results_df.loc[results_df['R2 Score'].idxmax(), 'Model']
    if best_model_name in trained_models:
        joblib.dump(trained_models[best_model_name], xgb_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, 'w') as f:
        json.dump(results_df.to_dict(orient='records'), f, indent=2)

    return trained_models, results_df, X_test_scaled, y_test, scaler
