import pandas as pd
import numpy as np

def create_features(df):
    """Feature engineering: rolling averages, lags, rates, temporal features."""
    df = df.sort_values(['country', 'date'])
    
    # 1. Rolling Averages (7-day)
    df['cases_7d_avg'] = df.groupby('country')['new_cases'].transform(lambda x: x.rolling(7).mean()).fillna(0)
    df['deaths_7d_avg'] = df.groupby('country')['new_deaths'].transform(lambda x: x.rolling(7).mean()).fillna(0)
    
    # 2. Daily Growth Percentage
    df['growth_rate'] = df.groupby('country')['new_cases'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Mortality Rate
    df['mortality_rate'] = (df['total_deaths'] / df['total_cases']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 4. Temporal features
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    
    # 5. Lag features
    df['lag_1'] = df.groupby('country')['new_cases'].shift(1).fillna(0)
    df['lag_7'] = df.groupby('country')['new_cases'].shift(7).fillna(0)
    
    return df
