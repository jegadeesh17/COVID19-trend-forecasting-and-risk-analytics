import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(df, comparison_df, y_test, predictions, feature_importance, output_dir='../visualizations'):
    """Generate and save project visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Heatmap
    plt.figure(figsize=(12, 10))
    cols = ['new_cases', 'total_cases', 'new_deaths', 'total_deaths', 
            'cases_7d_avg', 'deaths_7d_avg', 'growth_rate', 'mortality_rate', 'lag_1', 'lag_7']
    # Filter columns that exist in the dataframe
    corr_cols = [c for c in cols if c in df.columns]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Key Features')
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 2. Trend Plot
    plt.figure(figsize=(14, 6))
    global_trend = df.groupby('date')[['new_cases', 'cases_7d_avg']].sum().reset_index()
    plt.plot(global_trend['date'], global_trend['new_cases'], alpha=0.3, label='Daily Cases')
    plt.plot(global_trend['date'], global_trend['cases_7d_avg'], color='red', linewidth=2, label='7-Day MA')
    plt.title('Global COVID-19 Trend (Actual vs Moving Average)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "trend_analysis.png"))
    plt.close()
    
    # 3. Model Comparison
    if comparison_df is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='R2 Score', data=comparison_df, palette='viridis')
        plt.title('Model Performance Comparison (R² Score)')
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(output_dir, "model_comparison.png"))
        plt.close()
    
    # 4. Feature Importance
    if feature_importance is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='magma')
        plt.title('Top 10 Feature Importance (XGBoost)')
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()
