import streamlit as st
import pandas as pd
from services import load_and_prep_data
from components.cards import display_kpi_row
from components.visualizations import render_trend_chart

st.title("🌍 Global Health Intelligence & Infection Trajectories")
st.markdown("Analyze comprehensive worldwide pandemic progression, historical moving averages, and local infection severity metrics.")

# Pull cached data securely
df = load_and_prep_data()

# Read active context state from sidebar routing
selected_country = st.session_state.get("selected_country", "Global")

st.subheader(f"📊 Statistical Aggregate Summary: **{selected_country}**")

# Construct tailored regional views based on filter parameters
if selected_country == "Global":
    # Aggregate counts across global timeline
    view_df = df.groupby('date')[['new_cases', 'total_cases', 'new_deaths', 'total_deaths', 'cases_7d_avg']].sum().reset_index()
    latest = view_df.iloc[-1]
    
    # Calculate robust macro rates
    growth_pct = df.groupby('date')['growth_rate'].mean().iloc[-1] * 100
    mortality = (latest['total_deaths'] / latest['total_cases']) * 100 if latest['total_cases'] > 0 else 0.0
else:
    # Filter specific single entity
    view_df = df[df['country'] == selected_country].sort_values('date')
    if len(view_df) > 0:
        latest = view_df.iloc[-1]
        growth_pct = latest['growth_rate'] * 100
        mortality = latest['mortality_rate'] * 100
    else:
        st.warning(f"No valid historical data matches region: {selected_country}")
        st.stop()

# Populate custom UI metric blocks
display_kpi_row(
    total_cases=latest['total_cases'],
    daily_cases=latest['new_cases'],
    mortality_pct=mortality,
    growth_pct=growth_pct
)

st.divider()

# Render advanced dual-trace interactive Plotly wrappers
chart_title = f"Historical Infection Wave Dynamics ({selected_country})"
st.plotly_chart(
    render_trend_chart(view_df, title=chart_title), 
    width='stretch'
)

# Render auxiliary data context table
with st.expander("📁 View Comprehensive Granular Datasets"):
    st.dataframe(
        view_df.sort_values('date', ascending=False).head(100),
        width='stretch'
    )
