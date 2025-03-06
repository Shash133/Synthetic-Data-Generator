import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, path):
    """Save a PyTorch model"""
    torch.save(model.state_dict(), path)
    
def load_model(model, path):
    """Load a PyTorch model"""
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def plot_losses(g_losses, d_losses):
    """Plot generator and discriminator losses"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(g_losses, label='Generator')
    ax.plot(d_losses, label='Discriminator')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    return fig

def create_synthetic_dataframe(features, target, feature_names, target_name='quality'):
    """Create a pandas DataFrame from synthetic data"""
    df = pd.DataFrame(features, columns=feature_names)
    df[target_name] = target
    return df
