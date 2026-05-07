import pandas as pd
import numpy as np

def preprocess_data(df):
    """Handle missing values, duplicates, and standardization."""
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values: Interpolate numeric columns grouped by country
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby('country')[numeric_cols].transform(lambda x: x.interpolate().ffill().bfill().fillna(0))
    
    # Outlier handling (IQR) for key target variables
    for col in ['new_cases', 'new_deaths']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        
    return df
