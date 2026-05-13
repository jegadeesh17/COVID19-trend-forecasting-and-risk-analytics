import os
import streamlit as st
import pandas as pd
from config import DATA_PATH, FEATURES, TARGET
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.modeling import train_models

@st.cache_data(show_spinner="Loading and preparing global health records...")
def load_and_prep_data():
    """Load data bundle, interpolate missing points, and construct dynamic lag features."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file missing at requested path: {DATA_PATH}")
        
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = preprocess_data(raw_df)
    engineered_df = create_features(processed_df)
    return engineered_df

@st.cache_resource(show_spinner="Fitting optimized ML algorithms on focused historical trends...")
def load_cached_models():
    """Train regression pipelines over a highly lightweight subset to prevent cloud memory limits."""
    df = load_and_prep_data()
    
    # Extract feature inputs and target vector cleanly
    X = df[FEATURES]
    y = df[TARGET]
    
    # Strictly limit sample volume to 5,000 rows to guarantee instant Streamlit Cloud execution speeds
    # and avoid multi-threaded out-of-memory kernel freezes during cross-validation loops.
    if len(df) > 5000:
        sampled = df.sample(5000, random_state=42)
        X = sampled[FEATURES]
        y = sampled[TARGET]
        
    trained_models, results_df, X_test_scaled, y_test, scaler = train_models(X, y)
    return trained_models, results_df, X_test_scaled, y_test, scaler
