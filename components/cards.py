import streamlit as st

def display_kpi_row(total_cases, daily_cases, mortality_pct, growth_pct):
    """Renders highly styled row of metrics containers."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Confirmed Cases", 
            value=f"{total_cases:,.0f}"
        )
    with col2:
        st.metric(
            label="Latest Daily Cases", 
            value=f"{daily_cases:,.0f}"
        )
    with col3:
        st.metric(
            label="Mortality Ratio", 
            value=f"{mortality_pct:.2f}%",
            delta=f"{mortality_pct:+,.2f}% vs global",
            delta_color="inverse" if mortality_pct > 2.0 else "normal"
        )
    with col4:
        st.metric(
            label="Daily Growth Trend", 
            value=f"{growth_pct:.2f}%", 
            delta=f"{growth_pct:+,.2f}%", 
            delta_color="inverse"
        )
