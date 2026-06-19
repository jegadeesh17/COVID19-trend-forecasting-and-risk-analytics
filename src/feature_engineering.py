import pandas as pd
import numpy as np

def create_features(df, drop_warmup=True):
    """Feature engineering: rolling averages, lags, rates, temporal features.

    Args:
        df: Preprocessed DataFrame with 'country', 'date', 'new_cases', etc.
        drop_warmup: If True, drop initial rows per country where
                     rolling/lag features cannot be properly computed
                     instead of filling them with misleading zeros.
    """
    df = df.sort_values(['country', 'date'])

    # 1. Rolling Averages (7-day) - shifted by 1 to prevent lookahead bias
    df['cases_7d_avg'] = df.groupby('country')['new_cases'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df['deaths_7d_avg'] = df.groupby('country')['new_deaths'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    # 2. Daily Growth Percentage - shifted by 1
    df['growth_rate'] = df.groupby('country')['new_cases'].transform(
        lambda x: x.pct_change().shift(1)
    ).replace([np.inf, -np.inf], 0)

    # 3. Mortality Rate - uses shifted totals as cumulative totals at the end of the day include new cases
    df['total_cases_lag1'] = df.groupby('country')['total_cases'].shift(1).fillna(0)
    df['total_deaths_lag1'] = df.groupby('country')['total_deaths'].shift(1).fillna(0)
    df['mortality_rate'] = (df['total_deaths_lag1'] / df['total_cases_lag1']).replace([np.inf, -np.inf], 0).fillna(0)

    # 4. Temporal features
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype(int)

    # 5. Lag features
    df['lag_1'] = df.groupby('country')['new_cases'].shift(1)
    df['lag_7'] = df.groupby('country')['new_cases'].shift(7)

    # 6. Handle warmup period where rolling/lag features are undefined
    if drop_warmup:
        # Drop rows where any rolling/lag feature is NaN (first ~8 rows per country)
        warmup_cols = ['cases_7d_avg', 'deaths_7d_avg', 'lag_1', 'lag_7']
        df = df.dropna(subset=warmup_cols)
        # Fill any remaining NaN in growth_rate (e.g., from pct_change on zero values)
        df['growth_rate'] = df['growth_rate'].fillna(0)
    else:
        # Legacy fallback: fill all NaNs with 0 (less accurate but compatible)
        fill_cols = ['cases_7d_avg', 'deaths_7d_avg', 'growth_rate',
                     'lag_1', 'lag_7']
        for col in fill_cols:
            df[col] = df[col].fillna(0)

    return df
