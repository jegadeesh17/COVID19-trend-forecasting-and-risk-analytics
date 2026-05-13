import streamlit as st

def display_kpi_row(total_cases, mortality_pct):
    """Renders clean, uncluttered two-column metric overview focusing strictly on core non-zero statistical aggregates."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Total Confirmed Cases", 
            value=f"{total_cases:,.0f}"
        )
    with col2:
        st.metric(
            label="Mortality Ratio", 
            value=f"{mortality_pct:.2f}%",
            delta="Aggregate impact ratio",
            delta_color="off"
        )
