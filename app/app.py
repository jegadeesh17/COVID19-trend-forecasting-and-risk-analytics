import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

# Must be the first executed statement
st.set_page_config(
    page_title="COVID-19 Predictive Analytics",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import BASE_DIR, FEATURES
from src.services import load_and_prep_data, load_cached_models
from components.cards import display_kpi_row
from components.visualizations import render_trend_chart, render_forecast_chart, render_model_comparison
from src.modeling import forecast_global

def inject_custom_css():
    """Inject styling parameters for customized glassmorphism metric containers."""
    css_path = os.path.join(BASE_DIR, "app", "assets", "custom_style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    inject_custom_css()
    
    # Header area
    st.title("🦠 COVID-19 Predictive Analytics System")
    st.caption("Global Outbreak Intelligence, Time-Series Smoothing, & Machine Learning Forecasting Platform")
    
    # Execute primary caching steps securely
    try:
        df = load_and_prep_data()
    except Exception as e:
        st.error(f"Critical error ingesting historical bundle: {e}")
        st.stop()
        
    # Configure global sidebar navigation parameters
    st.sidebar.title("⚙️ Engine Controls")
    
    # Safe regional selector list
    countries = ["Global"] + sorted(list(df["country"].dropna().unique()))
    selected_country = st.sidebar.selectbox(
        "📍 Outbreak Region Focus", 
        options=countries, 
        index=0
    )
    
    # Safe horizon slider
    forecast_days = st.sidebar.slider(
        "Forecasting Horizon (Days)", 
        min_value=7, 
        max_value=60, 
        value=14, 
        step=1
    )
    
    st.sidebar.divider()
    st.sidebar.info("💡 **Tip:** Switch tabs below to navigate smoothly across aggregated health trends, projected infection surges, and algorithm feature matrices.")

    # Initialize intuitive native multi-tab UI
    tab_dash, tab_forecast, tab_eval = st.tabs([
        "🌍 Real-Time Outbreak Dashboard", 
        "📈 ML Trend Forecasting Engine", 
        "🧪 Diagnostics & Feature Analytics"
    ])
    
    # ==========================================
    # TAB 1: Real-Time Outbreak Dashboard
    # ==========================================
    with tab_dash:
        st.subheader(f"📊 Outbreak Health Snapshot: **{selected_country}**")
        
        # Determine specific frame aggregations based on global context
        if selected_country == "Global":
            view_df = df.groupby('date')[['new_cases', 'total_cases', 'new_deaths', 'total_deaths', 'cases_7d_avg']].sum().reset_index()
            latest = view_df.iloc[-1]
            growth_pct = df.groupby('date')['growth_rate'].mean().iloc[-1] * 100
            mortality = (latest['total_deaths'] / latest['total_cases']) * 100 if latest['total_cases'] > 0 else 0.0
        else:
            view_df = df[df['country'] == selected_country].sort_values('date')
            latest = view_df.iloc[-1]
            growth_pct = latest['growth_rate'] * 100
            mortality = latest['mortality_rate'] * 100
            
        # Display metric block array
        display_kpi_row(
            total_cases=latest['total_cases'],
            mortality_pct=mortality
        )
        
        st.divider()
        
        # Interactive trajectory display using width natively
        st.plotly_chart(
            render_trend_chart(view_df, title=f"Historical Case Tracking ({selected_country})"), 
            width="stretch"
        )
        
        with st.expander("📁 Inspect Raw Data Snapshot"):
            st.dataframe(view_df.tail(50), width="stretch")

    # ==========================================
    # TAB 2: ML Trend Forecasting Engine
    # ==========================================
    with tab_forecast:
        st.subheader(f"🔮 Recursive Forward Multi-Day Simulations")
        
        # Safely pull memory cached fitted algorithm pipelines
        with st.spinner("Initializing models backend..."):
            trained_models, results_df, X_test_scaled, y_test, scaler = load_cached_models()
            
        model_opts = list(trained_models.keys())
        active_model = st.selectbox(
            "🧠 Active Regressor Architecture", 
            options=model_opts, 
            index=model_opts.index("XGBoost") if "XGBoost" in model_opts else 0
        )
        
        # Execute recursive iteration
        forecast_res = forecast_global(
            model=trained_models[active_model],
            df=df,
            features=FEATURES,
            scaler=scaler,
            days=forecast_days
        )
        
        # Construct connected visualization historical bounds
        hist_agg = df.groupby('date')['new_cases'].sum().reset_index().sort_values('date')
        hist_tail = hist_agg.tail(60)
        
        st.plotly_chart(
            render_forecast_chart(
                historical_dates=hist_tail['date'],
                historical_values=hist_tail['new_cases'],
                forecast_dates=forecast_res['Date'],
                forecast_values=forecast_res['Predicted_New_Cases'],
                model_name=active_model
            ),
            width="stretch"
        )
        
        # Key statistical metrics breakdown
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Expected Peak Infection Surge", f"{forecast_res['Predicted_New_Cases'].max():,.0f} cases")
        with c2:
            st.metric("Average Trajectory Velocity", f"{forecast_res['Predicted_New_Cases'].mean():,.0f} cases/day")

    # ==========================================
    # TAB 3: Diagnostics & Feature Analytics
    # ==========================================
    with tab_eval:
        st.subheader("🧮 Model Benchmarking Comparison")
        st.dataframe(results_df[['Model', 'R2 Score', 'RMSE', 'MAE']], width="stretch", hide_index=True)
        
        st.plotly_chart(render_model_comparison(results_df), width="stretch")
        
        st.divider()
        st.subheader("🔑 Explanatory Feature Importance Weights (XGBoost)")
        
        if "XGBoost" in trained_models and hasattr(trained_models["XGBoost"], "feature_importances_"):
            imp_df = pd.DataFrame({
                'Feature': FEATURES,
                'Importance': trained_models["XGBoost"].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                imp_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                color='Importance',
                color_continuous_scale='Magma',
                title="Relative Impact of Epidemiological Variables on Case Growth"
            )
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width="stretch")

if __name__ == "__main__":
    main()
