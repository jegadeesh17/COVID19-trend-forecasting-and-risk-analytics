import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_models(X, y, dates=None):
    """Train and evaluate multiple regression models with time-series-aware splitting.

    Args:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
        dates: Optional date Series for chronological sorting.  If provided,
               data is sorted by date before splitting to ensure the train/test
               boundary is temporal, not country-based.

    Returns:
        trained_models: dict of fitted model objects
        results_df: DataFrame with evaluation metrics per model
        X_test_scaled: scaled test features (ndarray)
        y_test: test target values (Series)
        scaler: fitted StandardScaler
    """
    # Sort by date to ensure chronological train/test split across all countries
    if dates is not None:
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y.iloc[sort_idx].reset_index(drop=True)

    # Chronological 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
    }

    # Time-series-aware cross-validation (no future data leakage)
    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    trained_models = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        results.append({
            'Model': name,
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'R2 Score': r2_score(y_test, preds),
            'CV R2 Mean': cv_scores.mean()
        })
        trained_models[name] = model

    return trained_models, pd.DataFrame(results), X_test_scaled, y_test, scaler


def forecast_global(model, df, features, scaler, days=14):
    """Generate a multi-step global forecast via recursive prediction.

    Predicts for the 'average' country using the most recent global feature
    averages, then scales up by the number of active countries.  Features are
    updated at each step to reflect the latest predictions.
    """
    last_date = df['date'].max()
    latest_global = df[df['date'] == last_date][features].mean().to_frame().T

    num_countries = df['country'].nunique()

    forecast_values = []
    dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    current_input = latest_global.copy()

    # Seed recent history for rolling-average updates
    recent_preds = list(df.groupby('date')['new_cases'].sum().tail(7).values)

    for i in range(days):
        scaled_input = scaler.transform(current_input)

        # Predict for the "average" country, then scale to global level
        pred = max(0, model.predict(scaled_input)[0]) * num_countries

        forecast_values.append(pred)
        recent_preds.append(pred)
        per_country_pred = pred / num_countries

        # --- Update features for the next recursive step ---
        current_input['total_cases_lag1'] += per_country_pred

        # Update 7-day rolling average with recent predictions
        current_input['cases_7d_avg'] = np.mean(recent_preds[-7:]) / num_countries

        # Update growth rate from consecutive global predictions
        if len(forecast_values) >= 2 and forecast_values[-2] > 0:
            current_input['growth_rate'] = (
                (forecast_values[-1] - forecast_values[-2]) / forecast_values[-2]
            )
        else:
            current_input['growth_rate'] = 0.0

        current_input['lag_7'] = current_input['lag_1']
        current_input['lag_1'] = per_country_pred
        current_input['month'] = dates[i].month
        current_input['week'] = dates[i].isocalendar()[1]

    return pd.DataFrame({'Date': dates, 'Predicted_New_Cases': forecast_values})
