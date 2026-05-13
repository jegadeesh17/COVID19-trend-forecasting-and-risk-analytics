import streamlit as st
import pandas as pd
import plotly.express as px
from services import load_and_prep_data

st.title("🗺️ Regional Outbreak Drill-down & Correlation Analysis")
st.markdown("Perform granular localized behavioral studies and explore feature interdependence via dynamic correlation matrices.")

# Extract stabilized cached global sets
df = load_and_prep_data()
selected_country = st.session_state.get("selected_country", "Global")

st.subheader(f"📍 Geographic Entity Focus: **{selected_country}**")

# Isolate regional subsets
if selected_country == "Global":
    reg_df = df
else:
    reg_df = df[df['country'] == selected_country]

if len(reg_df) == 0:
    st.warning(f"Insufficient active history to compute correlations for region: {selected_country}")
    st.stop()

# Layout interactive country specific scatter trends
st.markdown("#### Dynamic Feature Interdependence")
col1, col2 = st.columns(2)
numeric_features = ['new_cases', 'total_cases', 'new_deaths', 'total_deaths', 'cases_7d_avg', 'growth_rate', 'mortality_rate']
valid_features = [c for c in numeric_features if c in reg_df.columns]

with col1:
    x_axis = st.selectbox("X-Axis Metric", options=valid_features, index=0)
with col2:
    y_axis = st.selectbox("Y-Axis Metric", options=valid_features, index=valid_features.index('new_deaths') if 'new_deaths' in valid_features else 0)

scatter_fig = px.scatter(
    reg_df, 
    x=x_axis, 
    y=y_axis, 
    color='month',
    title=f"Scatter Distribution: {x_axis} vs {y_axis}",
    template='plotly_dark',
    color_continuous_scale='Blues'
)
scatter_fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(scatter_fig, use_container_width=True)

st.divider()

# Compute exact correlation matrices
st.subheader("🧮 Interactive Correlation Matrix Heatmap")
corr_cols = [c for c in valid_features + ['lag_1', 'lag_7'] if c in reg_df.columns]
corr_matrix = reg_df[corr_cols].corr()

heatmap_fig = px.imshow(
    corr_matrix,
    text_auto='.2f',
    aspect="auto",
    color_continuous_scale='RdBu_r',
    origin='lower',
    title=f"Feature Correlation Matrix ({selected_country})"
)
heatmap_fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(heatmap_fig, use_container_width=True)
