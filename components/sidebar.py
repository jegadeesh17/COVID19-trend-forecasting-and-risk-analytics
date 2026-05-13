import streamlit as st
import pandas as pd
from config import APP_TITLE, APP_SUBTITLE

def render_sidebar(df: pd.DataFrame = None):
    st.sidebar.title(f"🦠 {APP_TITLE}")
    st.sidebar.caption(APP_SUBTITLE)
    st.sidebar.divider()
    
    # Store global region filter in Session State
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = "Global"
        
    if df is not None and "country" in df.columns:
        # Populate sidebar selectbox
        countries = ["Global"] + sorted(list(df["country"].dropna().unique()))
        
        # Determine safe current index
        try:
            curr_index = countries.index(st.session_state.selected_country)
        except ValueError:
            curr_index = 0
            
        selected = st.sidebar.selectbox(
            "📍 Select Outbreak Region", 
            options=countries,
            index=curr_index,
            help="Filter analytics across the entire global aggregation or zoom into specific regional metrics."
        )
        st.session_state.selected_country = selected
        
    st.sidebar.subheader("⚙️ Forecasting Engine Parameters")
    forecast_window = st.sidebar.slider(
        "Horizon Projection (Days)", 
        min_value=7, 
        max_value=60, 
        value=14, 
        step=1,
        help="Select the recursive projection window to simulate upcoming infection surge paths."
    )
    st.session_state.forecast_window = forecast_window
    
    st.sidebar.divider()
    st.sidebar.info("🤖 **Models Configured:** Linear Regression, Random Forest, & XGBoost with Cross-Validation validation.")
