import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp, wasserstein_distance, entropy

def create_comparison_plots(real_df, synth_df):
    """Create plots comparing real and synthetic data"""
    # Add data type column
    real_df_copy = real_df.copy()
    synth_df_copy = synth_df.copy()
    real_df_copy['data_type'] = 'Real'
    synth_df_copy['data_type'] = 'Synthetic'
    combined_df = pd.concat([real_df_copy, synth_df_copy])
    
    # Feature distributions
    fig1 = make_subplots(rows=3, cols=4, subplot_titles=real_df.columns.tolist())
    
    row, col = 1, 1
    for feature in real_df.columns:
        fig1.add_trace(
            go.Histogram(x=real_df[feature], name='Real', opacity=0.7,
                         marker_color='blue', nbinsx=20),
            row=row, col=col
        )
        fig1.add_trace(
            go.Histogram(x=synth_df[feature], name='Synthetic', opacity=0.7,
                         marker_color='red', nbinsx=20),
            row=row, col=col
        )
        
        col += 1
        if col > 4:
            col = 1
            row += 1
    
    fig1.update_layout(
        height=900, width=1000,
        title_text="Feature Distributions: Real vs Synthetic",
        showlegend=True,
        barmode='overlay'
    )
    
    # Correlation matrices
    fig2 = make_subplots(rows=1, cols=2,
                        subplot_titles=['Real Data Correlation', 'Synthetic Data Correlation'])
    
    real_corr = real_df.corr()
    synth_corr = synth_df.corr()
    
    fig2.add_trace(
        go.Heatmap(z=real_corr.values, x=real_corr.columns, y=real_corr.columns,
                  colorscale='RdBu_r', zmin=-1, zmax=1),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Heatmap(z=synth_corr.values, x=synth_corr.columns, y=synth_corr.columns,
                  colorscale='RdBu_r', zmin=-1, zmax=1),
        row=1, col=2
    )
    
    fig2.update_layout(height=600, width=1200, title_text="Correlation Matrix Comparison")
    
    return fig1, fig2

def calculate_metrics(real_df, synth_df):
    """Calculate statistical metrics comparing real and synthetic data"""
    features = real_df.columns[:-1]  # Exclude quality for continuous features
    
    # KS test
    ks_results = {}
    for col in features:
        stat, p_value = ks_2samp(real_df[col], synth_df[col])
        ks_results[col] = {'KS Statistic': stat, 'p-value': p_value}
    
    ks_df = pd.DataFrame(ks_results).T
    ks_df['Significantly Different'] = ks_df['p-value'] < 0.05
    
    # Wasserstein distance
    wasserstein_results = {}
    for feature in features:
        real_data = real_df[feature].values
        synth_data = synth_df[feature].values
        wasserstein_dist = wasserstein_distance(real_data, synth_data)
        wasserstein_results[feature] = wasserstein_dist
    
    # KL Divergence
    kl_div_results = {}
    for feature in features:
        real_data = real_df[feature].values
        synth_data = synth_df[feature].values
        
        # Create histograms (probability distributions)
        real_hist, _ = np.histogram(real_data, bins=20, density=True)
        synth_hist, _ = np.histogram(synth_data, bins=20, density=True)
        
        # Calculate KL Divergence (add small value to avoid log(0))
        kl_div = entropy(real_hist + 1e-10, synth_hist + 1e-10)
        kl_div_results[feature] = kl_div
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'KS Statistic': [ks_results[f]['KS Statistic'] for f in features],
        'KS p-value': [ks_results[f]['p-value'] for f in features],
        'Wasserstein Distance': [wasserstein_results[f] for f in features],
        'KL Divergence': [kl_div_results[f] for f in features]
    }, index=features)
    
    return metrics_df

def plot_metrics(real_data, synthetic_data):
    """Create plots for evaluation metrics"""
    # KS Statistics
    metrics_df = calculate_metrics(real_data, synthetic_data)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=metrics_df.index,
        y=metrics_df['KS Statistic'],
        name='KS Statistic'
    ))
    fig1.add_shape(
        type="line",
        x0=-0.5, y0=0.1, x1=len(metrics_df)-0.5, y1=0.1,
        line=dict(color="red", width=2, dash="dash")
    )
    fig1.update_layout(
        title="Kolmogorov-Smirnov Statistics by Feature",
        xaxis_title="Feature",
        yaxis_title="KS Statistic",
        height=500
    )
    
    # Wasserstein Distance
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=metrics_df.index,
        y=metrics_df['Wasserstein Distance'],
        name='Wasserstein Distance'
    ))
    fig2.update_layout(
        title="Wasserstein Distance between Real and Synthetic Data",
        xaxis_title="Feature",
        yaxis_title="Wasserstein Distance",
        height=500
    )
        # KL Divergence metrics plot
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=metrics_df.index,
        y=metrics_df['KL Divergence'],
        name='KL Divergence'
    ))
    fig3.update_layout(
        title="KL Divergence between Real and Synthetic Data",
        xaxis_title="Feature",
        yaxis_title="KL Divergence",
        height=500
    )
    
    return fig1, fig2, fig3, metrics_df
