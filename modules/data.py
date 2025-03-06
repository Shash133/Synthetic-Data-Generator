import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(config, return_plots=False):
    """
    Load and preprocess wine quality data
    
    Args:
        config: Configuration object
        return_plots: Whether to return plot figures
    
    Returns:
        Tuple of processed data and related objects
    """
    df = pd.read_csv(config.DATA_URL, sep=';')
    
    # Handle sample size
    if len(df) < config.SAMPLE_SIZE:
        df = df.sample(n=config.SAMPLE_SIZE, replace=True, random_state=42)
    else:
        df = df.sample(n=config.SAMPLE_SIZE, random_state=42)
    
    # Separate features and target
    features = df.drop(config.TARGET_COL, axis=1)
    target = df[[config.TARGET_COL]]
    
    # Store original ranges
    value_ranges = {
        'min': features.min().values,
        'max': features.max().values,
        'positive': [True]*features.shape[1]
    }
    
    # Scale features
    num_scaler = MinMaxScaler(feature_range=config.MIN_MAX_RANGE)
    scaled_features = num_scaler.fit_transform(features)
    
    # Encode target
    cat_encoder = OneHotEncoder(sparse_output=False)
    encoded_target = cat_encoder.fit_transform(target)
    
    # Combine features and target
    combined_data = np.hstack([scaled_features, encoded_target])
    
    # Create plots if requested
    plots = None
    if return_plots:
        plots = create_data_plots(df)
    
    return (combined_data, num_scaler, cat_encoder, features, target, df, 
            value_ranges, plots)

def create_data_plots(df):
    """Create exploratory plots for the dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution of quality
    sns.countplot(x='quality', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Wine Quality')
    
    # Correlation heatmap
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, ax=axes[0, 1])
    axes[0, 1].set_title('Feature Correlation Heatmap')
    
    # Feature distributions
    df.drop('quality', axis=1).boxplot(ax=axes[1, 0])
    axes[1, 0].set_title('Feature Distributions')
    axes[1, 0].tick_params(axis='x', rotation=90)
    
    # Alcohol vs Quality
    sns.boxplot(x='quality', y='alcohol', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Alcohol Content by Quality')
    
    plt.tight_layout()
    return fig
