import pandas as pd
import numpy as np

def preprocess_data(df):
    """Handle missing values, duplicates, and standardization."""
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')

    # Convert date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values: Forward fill numeric columns grouped by country to prevent lookahead bias
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby('country')[numeric_cols].ffill().fillna(0)

    # Memory Downcasting
    for col in ['country', 'iso_code']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Selectively downcast — skip columns with large values to preserve precision
    float_cols = df.select_dtypes(include=['float64']).columns
    safe_cols = [c for c in float_cols if df[c].abs().max() < 1e7]
    if safe_cols:
        df[safe_cols] = df[safe_cols].astype('float32')

    return df
