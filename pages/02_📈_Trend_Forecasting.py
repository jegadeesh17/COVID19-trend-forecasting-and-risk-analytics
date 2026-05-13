import streamlit as st
import pandas as pd
from datetime import timedelta
from services import load_and_prep_data, load_cached_models
from config import FEATURES
from src.modeling import forecast_global
from components.visualizations import render_forecast_chart

st.title("📈 Machine Learning Outbreak Projections")
st.markdown("Execute advanced recursive time-series forecasting to simulate upcoming pandemic propagation paths based on trained ensemble models.")

# Pull persistent shared memory datasets and trained scikit-learn/XGBoost structures
df = load_and_prep_data()
trained_models, results_df, X_test_scaled, y_test, scaler = load_cached_models()

# Layout model tuning configuration controls
col1, col2 = st.columns(2)
with col1:
    model_options = list(trained_models.keys())
    selected_model = st.selectbox(
        "🧠 Select Active Regression Architecture", 
        options=model_options,
        index=model_options.index("XGBoost") if "XGBoost" in model_options else 0,
        help="Choose the mathematical structure driving the recursive multi-day feedback simulation."
    )
    
with col2:
    # Read active slider state or provide secondary tuning option
    current_window = st.session_state.get("forecast_window", 14)
    forecast_days = st.number_input(
        "Simulation Projection Window (Days)", 
        min_value=7, 
        max_value=90, 
        value=current_window,
        step=1,
        help="Determines how far into the unobserved future the recursive algorithm shifts inputs."
    )
    # Sync parameter back to session
    st.session_state.forecast_window = forecast_days

# Execute forecasting engine
with st.spinner(f"Simulating forward trajectory using {selected_model}..."):
    forecast_res = forecast_global(
        model=trained_models[selected_model],
        df=df,
        features=FEATURES,
        scaler=scaler,
        days=forecast_days
    )

# Extract historical context line leading up to projection threshold
st.subheader(f"🔮 {forecast_days}-Day Infection Trajectory Simulation")

# Compute global past aggregation over the final 60 observed days
hist_df = df.groupby('date')['new_cases'].sum().reset_index().sort_values('date')
recent_hist = hist_df.tail(60)

# Connect historical values to forecast seamlessly
st.plotly_chart(
    render_forecast_chart(
        historical_dates=recent_hist['date'],
        historical_values=recent_hist['new_cases'],
        forecast_dates=forecast_res['Date'],
        forecast_values=forecast_res['Predicted_New_Cases'],
        model_name=selected_model
    ),
    use_container_width=True
)

# Render side-by-side metric breakdown
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="Forecast Start Baseline", 
        value=f"{recent_hist.iloc[-1]['new_cases']:,.0f} cases"
    )
with c2:
    peak_pred = forecast_res['Predicted_New_Cases'].max()
    st.metric(
        label="Projected Surge Peak", 
        value=f"{peak_pred:,.0f} cases",
        delta=f"{peak_pred - recent_hist.iloc[-1]['new_cases']:+,.0f} vs base",
        delta_color="inverse"
    )
with c3:
    avg_pred = forecast_res['Predicted_New_Cases'].mean()
    st.metric(
        label="Average Horizon Velocity", 
        value=f"{avg_pred:,.0f} cases/day"
    )

with st.expander("📄 Export Future Trajectory Records (CSV / Dataframe)"):
    st.dataframe(forecast_res, use_container_width=True)
