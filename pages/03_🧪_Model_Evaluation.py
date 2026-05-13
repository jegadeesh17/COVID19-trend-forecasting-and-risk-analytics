import streamlit as st
import pandas as pd
import plotly.express as px
from services import load_cached_models
from config import FEATURES
from components.visualizations import render_model_comparison

st.title("🧪 Regression Benchmarking & Model Analytics")
st.markdown("Evaluate institutional machine learning performance across regression algorithms using validation statistics and feature importance matrices.")

# Pull persistent execution models and evaluation outcomes
trained_models, results_df, X_test_scaled, y_test, scaler = load_cached_models()

st.subheader("📊 Comparative Validation Statistics")

# Display metrics overview cleanly
st.dataframe(
    results_df[['Model', 'R2 Score', 'CV R2 Mean', 'RMSE', 'MAE']],
    use_container_width=True,
    hide_index=True
)

st.divider()

# High-fidelity visual bar rendering
st.plotly_chart(
    render_model_comparison(results_df),
    use_container_width=True
)

# Render feature importance tracking
st.subheader("🔑 Feature Importance Weights (XGBoost Engine)")
if "XGBoost" in trained_models:
    xgb_model = trained_models["XGBoost"]
    if hasattr(xgb_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Top Feature Contributions to Infection Spread Velocity",
            color='Importance',
            color_continuous_scale='Magma'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance diagnostics require successful XGBoost fitting.")
