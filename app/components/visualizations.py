import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_trend_chart(df: pd.DataFrame, title="Outbreak Trajectory & Moving Averages"):
    """Render professional dual-axis trend analysis plot with transparent dark themes."""
    fig = go.Figure()
    
    # Actual daily cases area plot
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['new_cases'],
        mode='lines',
        name='Daily Cases',
        line=dict(color='rgba(59, 130, 246, 0.5)', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.08)'
    ))
    
    # Smooth 7-day rolling average trace
    if 'cases_7d_avg' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['cases_7d_avg'],
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='#EF4444', width=2.5)
        ))
        
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=50, b=20),
        hovermode="x unified",
        xaxis_title="Observation Date",
        yaxis_title="Confirmed Infection Counts"
    )
    return fig

def render_forecast_chart(historical_dates, historical_values, forecast_dates, forecast_values, model_name="Model"):
    """Render forecast curve projected after the historical threshold line."""
    fig = go.Figure()
    
    # Historical Trace
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_values,
        mode='lines',
        name='Historical Input State',
        line=dict(color='#3B82F6', width=2.0)
    ))
    
    # Forecast Projection Trace
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        mode='lines+markers',
        name=f'{model_name} Prediction Projection',
        line=dict(color='#10B981', width=2.5, dash='dot'),
        marker=dict(size=5)
    ))
    
    fig.update_layout(
        title=f"Recursive Time-Series Projection Trajectory ({model_name})",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=50, b=20),
        hovermode="x unified",
        xaxis_title="Date Horizon",
        yaxis_title="Predicted Case Counts"
    )
    return fig

def render_model_comparison(results_df: pd.DataFrame):
    """Render bar charts comparing regression scores across algorithms."""
    fig = px.bar(
        results_df, 
        x='Model', 
        y='R2 Score',
        color='R2 Score',
        color_continuous_scale='Viridis',
        text='R2 Score',
        title="Model Performance Comparison (R² Coefficient of Determination)"
    )
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_range=[0, 1.1]
    )
    return fig
