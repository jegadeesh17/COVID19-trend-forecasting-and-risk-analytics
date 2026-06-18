# Global COVID-19 Trend Forecasting and Risk Analytics

---

### **Project Overview**

The COVID-19 pandemic created unprecedented challenges in monitoring infection spread, predicting outbreak trends, and understanding regional risk behavior. This project builds an AI-driven predictive analytics platform using machine learning, time-series analytics, and data visualization to analyze the global pandemic's trajectory.

The system processes historical COVID-19 data, engineers advanced temporal features, forecasts future case trends using ensemble models, and visualizes outbreak patterns — all delivered through a multi-page Streamlit dashboard with glassmorphism UI aesthetics and high-fidelity Plotly charts.

---

### **Key Features**

* **Pandemic Trend Analytics:** Analyzes COVID-19 infection, recovery, and mortality trends over time.
* **Machine Learning Forecasting:** Predicts future case counts using Linear Regression, Random Forest, and XGBoost.
* **Advanced Feature Engineering:** Rolling averages, growth rates, mortality rates, recovery rates, and lag-based forecasting features.
* **Interactive Data Visualization:** Trend graphs, correlation heatmaps, Prediction vs Actual charts, and model comparison dashboards.
* **Modular ML Pipeline:** Separate preprocessing, feature engineering, modeling, evaluation, and visualization scripts.
* **Multi-Page Streamlit App:** Glassmorphism-styled dashboard with session state continuity across analytics views.
* **Model Performance Evaluation:** Compares all models using R², RMSE, and MAE metrics.
* **Automated Analytics Pipeline:** End-to-end preprocessing, forecasting, evaluation, and visualization workflows.

---

### **Dataset**

* **Source:** Global COVID-19 Dataset
* **Coverage:** Worldwide pandemic statistics by country and date
* **Data Type:** Time-series healthcare and epidemiological data

#### **Key Features**

* Confirmed cases count
* Death counts and mortality rates
* Recovery statistics
* Daily growth rate trends
* Regional outbreak information
* Date-wise pandemic observations (time-series)

---

### **Project Structure**

```bash
CovidPredictiveAnalytics/
│
├── app/                          # Streamlit application files
│   └── app.py                    # Main Streamlit dashboard
├── data/                         # Project datasets
├── docs/                         # Documentation and visualizations
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks (Source of Truth)
├── src/                          # Core Python logic and scripts
├── requirements.txt              # Python dependencies
└── README.md
```

---

### **How It Works**

### **1. Data Preprocessing**

* Cleans missing and inconsistent pandemic records
* Converts date columns into proper time-series format
* Removes duplicates and outliers
* Standardizes numerical features for model training

---

### **2. Feature Engineering**

The system creates advanced temporal analytical features:

| Feature              | Purpose                                    |
| -------------------- | ------------------------------------------ |
| `daily_growth_rate`  | Tracks infection growth trends             |
| `rolling_avg_cases`  | Smooths short-term fluctuations            |
| `mortality_rate`     | Measures outbreak severity                 |
| `recovery_rate`      | Evaluates healthcare recovery trends       |
| `lag_7`              | 7-day lagged cases for trend forecasting   |
| `lag_14`             | 14-day lagged cases for trend forecasting  |

---

### **3. Machine Learning Forecasting**

#### Models Used

* **Linear Regression** — Baseline trend forecasting
* **Random Forest Regressor** — Non-linear pattern capture
* **XGBoost Regressor** — High-accuracy ensemble forecasting

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
```

---

### **Model Performance**

| Metric   | Score           |
| -------- | --------------- |
| R² Score | 0.92            |
| RMSE     | Low Error Range |
| MAE      | Optimized       |

---

### **4. Forecasting Capabilities**

* Future case prediction using trained ensemble models
* Moving average trend continuation analysis
* Outbreak surge identification
* Multi-step ahead forecasting using lag features

---

### **Interactive Application Deployment**

The project features a fully institutionalized, multi-page **Streamlit Web Application** designed with glassmorphism CSS tokens, session state continuity, and high-fidelity interactive Plotly charts.

#### **To Launch the Platform Locally:**
```powershell
streamlit run app/app.py
```

---

### **Technology Stack**

| Category             | Tools                          |
| -------------------- | ------------------------------ |
| Programming          | Python                         |
| Data Processing      | Pandas, NumPy                  |
| Machine Learning     | Scikit-learn, XGBoost          |
| Visualization        | Matplotlib, Seaborn, Plotly    |
| Forecasting          | Time-Series Analytics          |
| Notebook Environment | Jupyter Notebook               |
| Web Framework        | Streamlit                      |

---

### **Getting Started**

### **1. Clone Repository**

```bash
git clone https://github.com/jegadeesh17/COVID19-Trend-Forecasting-and-Risk-Analytics.git

cd CovidPredictiveAnalytics
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **3. Launch Notebook**

```bash
jupyter notebook
```

Open:

```bash
notebooks/covid_analysis.ipynb
```

---

### **4. Train Models**

```python
from xgboost import XGBRegressor

model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
```

---

### **5. Launch Dashboard**

```bash
python -m streamlit run app.py
```

---

### **Example Use Case**

Healthcare analysts and policymakers can use this platform to:

1. Monitor real-time pandemic growth trends by country or region
2. Forecast future case surges to plan healthcare capacity
3. Analyze regional mortality and recovery rate disparities
4. Support data-driven public health policy decisions

---

### **Future Improvements**

* Real-time COVID-19 API integration for live data updates
* Deep learning-based forecasting using LSTM and Transformer models
* Geospatial outbreak mapping using Folium or Kepler.gl
* Vaccine distribution and efficacy analytics layer

---

### **Contributors**

* **Jegadeesh D** — Data preprocessing, time-series feature engineering, machine learning forecasting, evaluation, and Streamlit dashboard development

---

### **License**

MIT License
