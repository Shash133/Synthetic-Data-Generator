import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import time
import os
from modules.config import Config
from modules.data import load_and_preprocess_data
from modules.models import Generator, Discriminator
from modules.training import train_gan, generate_synthetic_samples
from modules.evaluation import create_comparison_plots, calculate_metrics, plot_metrics
from modules.utils import set_seeds, save_model, load_model, plot_losses, create_synthetic_dataframe

# Set page config
st.set_page_config(
    page_title="Wine Quality Synthetic Data Generator",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'trained_generator' not in st.session_state:
    st.session_state.trained_generator = None
if 'g_losses' not in st.session_state:
    st.session_state.g_losses = []
if 'd_losses' not in st.session_state:
    st.session_state.d_losses = []
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'real_data' not in st.session_state:
    st.session_state.real_data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'num_scaler' not in st.session_state:
    st.session_state.num_scaler = None
if 'value_ranges' not in st.session_state:
    st.session_state.value_ranges = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Create directory for models if it doesn't exist
os.makedirs('models', exist_ok=True)

# Title and introduction
st.title("🍷 Wine Quality Synthetic Data Generator")
st.markdown("""
This application uses a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate synthetic wine quality data.
The synthetic data maintains the statistical properties of the original dataset while providing new, diverse samples.
""")

# Sidebar
st.sidebar.header("Controls")

# Load data
if st.sidebar.button("Load Data") or st.session_state.data_loaded:
    with st.spinner("Loading and preprocessing data..."):
        config = Config()
        data_tensor, num_scaler, cat_encoder, features, target, df, value_ranges, plots = load_and_preprocess_data(config, return_plots=True)
        
        st.session_state.data_loaded = True
        st.session_state.data_tensor = data_tensor
        st.session_state.num_scaler = num_scaler
        st.session_state.value_ranges = value_ranges
        st.session_state.real_data = df
        st.session_state.feature_names = features.columns.tolist()
        
        st.success("Data loaded successfully!")
        
        # Display data exploration
        st.header("Data Exploration")
        st.pyplot(plots)
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of samples: {len(df)}")
            st.write(f"Number of features: {len(features.columns)}")
        with col2:
            st.write(f"Quality range: {df['quality'].min()} - {df['quality'].max()}")
            st.write(f"Most common quality: {df['quality'].mode()[0]}")
        
        # Display sample of the data
        st.subheader("Sample Data")
        st.dataframe(df.head())

# Train model section
st.sidebar.header("Model Training")
epochs = st.sidebar.slider("Number of Epochs", min_value=100, max_value=2000, value=500, step=100)
batch_size = st.sidebar.slider("Batch Size", min_value=64, max_value=1024, value=512, step=64)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    format_func=lambda x: f"{x:.5f}"
)

if st.sidebar.button("Train Model") and st.session_state.data_loaded:
    # Update config with user parameters
    config = Config()
    config.EPOCHS = epochs
    config.BATCH_SIZE = batch_size
    config.LR = learning_rate
    
    # Initialize models
    set_seeds(42)
    generator = Generator(config)
    discriminator = Discriminator(config)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Callback function to update progress
    def update_progress(epoch, total_epochs, g_loss, d_loss):
        progress = epoch / total_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch}/{total_epochs} - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
    
    # Train the model
    with st.spinner("Training the GAN model..."):
        g_losses, d_losses = train_gan(
            generator, discriminator, 
            st.session_state.data_tensor, 
            config,
            progress_callback=update_progress
        )
        
        # Save model
        save_model(generator, 'models/generator.pt')
        
        # Update session state
        st.session_state.trained_generator = generator
        st.session_state.g_losses = g_losses
        st.session_state.d_losses = d_losses
        
        progress_bar.progress(1.0)
        status_text.text("Training complete!")
        
    st.success("Model trained successfully!")
    
    # Plot losses
    st.subheader("Training Losses")
    fig = plot_losses(g_losses, d_losses)
    st.pyplot(fig)

# Generate synthetic data section
st.sidebar.header("Generate Synthetic Data")
num_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=5000, value=1000, step=100)

if st.sidebar.button("Generate Data"):
    # Check if model is trained or load pretrained model
    if st.session_state.trained_generator is None:
        try:
            config = Config()
            generator = Generator(config)
            generator = load_model(generator, 'models/generator.pt')
            st.session_state.trained_generator = generator
        except:
            st.error("No trained model found. Please train a model first.")
    
    if st.session_state.trained_generator is not None:
        with st.spinner("Generating synthetic data..."):
            # Generate synthetic samples
            synth_features, synth_target = generate_synthetic_samples(
                num_samples, 
                st.session_state.trained_generator,
                st.session_state.num_scaler,
                st.session_state.value_ranges,
                Config()
            )
            
            # Create DataFrame
            synth_df = create_synthetic_dataframe(
                synth_features, 
                synth_target, 
                st.session_state.feature_names
            )
            
            # Update session state
            st.session_state.synthetic_data = synth_df
            
        st.success(f"Generated {num_samples} synthetic wine samples!")
        
        # Display sample of synthetic data
        st.subheader("Synthetic Data Sample")
        st.dataframe(synth_df.head(10))
        
        # Download button for synthetic data
        csv = synth_df.to_csv(index=False)
        st.download_button(
            label="Download Synthetic Data as CSV",
            data=csv,
            file_name="synthetic_wine_quality.csv",
            mime="text/csv"
        )

# Data evaluation section
if st.session_state.synthetic_data is not None and st.session_state.real_data is not None:
    st.header("Data Evaluation")
    
    with st.spinner("Evaluating synthetic data quality..."):
        # Create comparison plots
        fig1, fig2, fig3, metrics_df = plot_metrics(
            st.session_state.real_data.drop('quality', axis=1), 
            st.session_state.synthetic_data.drop('quality', axis=1)
        )
        
        # Display distribution plots
        st.subheader("Feature Distributions: Real vs Synthetic")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Display correlation matrices
        st.subheader("Correlation Matrix Comparison")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display KL divergence
        st.subheader("KL Divergence between Real and Synthetic Data")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Display metrics table
        st.subheader("Statistical Metrics")
        st.dataframe(metrics_df)
        
        # Quality distribution comparison
        st.subheader("Quality Distribution Comparison")
        
        # Create quality distribution plot
        real_quality = st.session_state.real_data['quality'].value_counts(normalize=True).sort_index()
        synth_quality = st.session_state.synthetic_data['quality'].value_counts(normalize=True).sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=real_quality.index,
            y=real_quality.values,
            name='Real',
            marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=synth_quality.index,
            y=synth_quality.values,
            name='Synthetic',
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Quality Distribution: Real vs Synthetic",
            xaxis_title="Quality Score",
            yaxis_title="Proportion",
            barmode='group',
            height=500,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Feature exploration section
if st.session_state.synthetic_data is not None and st.session_state.real_data is not None:
    st.header("Feature Exploration")
    
    # Feature selector
    features = st.session_state.feature_names + ['quality']
    selected_feature = st.selectbox("Select a feature to explore:", features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Real Data - {selected_feature}")
        fig, ax = plt.subplots(figsize=(8, 6))
        if selected_feature == 'quality':
            ax.hist(st.session_state.real_data[selected_feature], bins=range(3, 10), alpha=0.7)
        else:
            ax.hist(st.session_state.real_data[selected_feature], bins=20, alpha=0.7)
        ax.set_title(f"Distribution of {selected_feature} (Real)")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    with col2:
        st.subheader(f"Synthetic Data - {selected_feature}")
        fig, ax = plt.subplots(figsize=(8, 6))
        if selected_feature == 'quality':
            ax.hist(st.session_state.synthetic_data[selected_feature], bins=range(3, 10), alpha=0.7)
        else:
            ax.hist(st.session_state.synthetic_data[selected_feature], bins=20, alpha=0.7)
        ax.set_title(f"Distribution of {selected_feature} (Synthetic)")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    # Feature relationship exploration
    st.subheader("Feature Relationships")
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis feature:", features, index=0)
    with col2:
        y_feature = st.selectbox("Select Y-axis feature:", features, index=1)
    
    if x_feature != y_feature:
        fig = go.Figure()
        
        # Add real data
        fig.add_trace(go.Scatter(
            x=st.session_state.real_data[x_feature],
            y=st.session_state.real_data[y_feature],
            mode='markers',
            name='Real',
            marker=dict(color='blue', opacity=0.5)
        ))
        
        # Add synthetic data
        fig.add_trace(go.Scatter(
            x=st.session_state.synthetic_data[x_feature],
            y=st.session_state.synthetic_data[y_feature],
            mode='markers',
            name='Synthetic',
            marker=dict(color='red', opacity=0.5)
        ))
        
        fig.update_layout(
            title=f"{y_feature} vs {x_feature}",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)

# About section
st.sidebar.header("About")
st.sidebar.info("""
This application demonstrates the use of Generative Adversarial Networks (GANs) 
for synthetic data generation. The model is trained on the UCI Wine Quality dataset 
and can generate new, synthetic wine samples that maintain the statistical properties 
of the original data.

Created with Streamlit and PyTorch.
""")

# Footer
st.markdown("---")
st.markdown("Wine Quality Synthetic Data Generator | Created with Streamlit and PyTorch")
