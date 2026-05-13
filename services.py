import os
import streamlit as st
import pandas as pd
from config import DATA_PATH, FEATURES, TARGET
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.modeling import train_models

@st.cache_data(show_spinner="Ingesting and stabilizing time-series datasets...")
def load_and_prep_data():
    """Load raw dataset, execute interpolation preprocessing, and extract advanced temporal lags."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Required global data bundle not found at path: {DATA_PATH}")
        
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_data(raw_df)
    engineered_df = create_features(processed_df)
    return engineered_df

@st.cache_resource(show_spinner="Training ensemble regressors (RandomForest, XGBoost) over sampled timeline...")
def load_cached_models():
    """Train machine learning pipelines on robust historical records and persist memory structures."""
    df = load_and_prep_data()
    
    # Extract feature inputs and target vector cleanly
    X = df[FEATURES]
    y = df[TARGET]
    
    # Downsample institutional data volume to preserve real-time Streamlit memory allocations
    if len(df) > 50000:
        sampled = df.sample(50000, random_state=42)
        X = sampled[FEATURES]
        y = sampled[TARGET]
        
    trained_models, results_df, X_test_scaled, y_test, scaler = train_models(X, y)
    return trained_models, results_df, X_test_scaled, y_test, scaler
