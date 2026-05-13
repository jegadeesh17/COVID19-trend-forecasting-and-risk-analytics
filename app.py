import os
import streamlit as st

# Application startup and theme binding MUST run before any secondary layouts
st.set_page_config(
    page_title="COVID-19 Predictive Analytics",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import BASE_DIR
from services import load_and_prep_data
from components.sidebar import render_sidebar

def inject_custom_css():
    """Inject vanilla CSS tokens to override standard spacing and border structures."""
    css_path = os.path.join(BASE_DIR, "assets", "custom_style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    inject_custom_css()
    
    # Trigger cached data load safely
    try:
        df = load_and_prep_data()
    except Exception as e:
        st.error(f"⚠️ Critical initialization failure during initial extraction pipeline: {e}")
        st.stop()
        
    # Render unified global filters on sidebar
    render_sidebar(df)
    
    # Instantiate navigation maps mapping paths to custom execution blocks
    pages = {
        "Global Pandemic Intelligence": [
            st.Page(os.path.join(BASE_DIR, "pages", "01_🌍_Global_Dashboard.py"), title="Global Health Dashboard", default=True),
            st.Page(os.path.join(BASE_DIR, "pages", "04_🗺️_Regional_Analytics.py"), title="Regional Drill-down Studies"),
        ],
        "Machine Learning Forecasting": [
            st.Page(os.path.join(BASE_DIR, "pages", "02_📈_Trend_Forecasting.py"), title="Time-Series Future Projections"),
            st.Page(os.path.join(BASE_DIR, "pages", "03_🧪_Model_Evaluation.py"), title="Regression Model Benchmarking"),
        ]
    }
    
    router = st.navigation(pages)
    router.run()

if __name__ == "__main__":
    main()
